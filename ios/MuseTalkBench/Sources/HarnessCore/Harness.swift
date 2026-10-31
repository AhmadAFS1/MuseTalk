import CoreML
import Foundation

public struct BenchOptions {
    /// Frames (not calls) in each UNet timing pass; the capture set is cycled.
    public var unetFrames = 200
    /// Compute units each UNet is timed and gated under.
    public var unetUnits: [MLComputeUnits] = [.all, .cpuAndNeuralEngine]
    /// Also gate the batch-1 UNet on CPU_AND_GPU (the accuracy anchor: exact on the Mac GPU). Slow on a phone.
    public var gpuCheck = true
    public var gpuCheckFrames = 64
    public var includeBatch4 = true
    /// Back-to-back batch-1 UNet calls under ALL for this long, bucketed by 5 s (thermal throttling).
    public var sustainedSeconds = 30.0
    /// UNet -> TAESD sequential frames.
    public var chainFrames = 100
    public var whisper = true
    /// Capture frames used for the gate (nil = all bundled frames).
    public var accuracyFrames: Int? = nil
    public var computePlan = true
    /// Load each UNet twice (first = compile/cache, second = cached figure). Off on memory-tight phones.
    public var reload = true
    /// Restrict the UNet runs to these names (nil = every bundled UNet; a chunked UNet is named `<parent>_chunked`).
    public var modelNames: [String]? = nil
    /// Skip the UNet phases altogether (Whisper only).
    public var unet = true
    /// TAESD compute units tried in the UNet -> TAESD chain (one chain run each).
    public var taesdUnits: [MLComputeUnits] = [.all]
    /// Pause after releasing a UNet before the next load (phones: the Neural Engine frees the previous program asynchronously).
    public var settleSeconds = 0.0
    /// Compute units for the UNet in the sustained and chain phases (ALL on the Mac; CPU_AND_NE on an A14, where ALL is 2x slower).
    public var phaseUnits: MLComputeUnits = .all

    public init() {}

    var record: OptionsRecord {
        OptionsRecord(unet_frames: unetFrames, unet_units: unetUnits.map(unitsName), gpu_check: gpuCheck,
                      gpu_check_frames: gpuCheckFrames, include_batch4: includeBatch4, sustained_seconds: sustainedSeconds,
                      chain_frames: chainFrames, whisper: whisper, reload: reload, settle_seconds: settleSeconds,
                      compute_plan: computePlan, phase_units: unitsName(phaseUnits))
    }
}

/// A UNet as the harness runs it: one Core ML model, or the chunks of one run back to back.
public struct UNetSpec {
    public let name: String
    public let batch: Int
    public let variant: String?
    public let layout: String?
    public let stages: [ModelMeta]

    /// The original inputs (sample / audio) are those of the first stage.
    var inputs: [IOFeature] { stages[0].inputs }
    var outputName: String { stages.last?.outputs.first?.name ?? "pred_latents" }
}

/// Loads a HarnessBundle folder produced by scripts/ios/prepare_harness_bundle.py and runs the benchmark.
public final class Harness {
    public let resources: URL
    public let meta: BundleMeta
    public var log: (String) -> Void = { print($0) }
    public var isCancelled: () -> Bool = { false }

    private let latents: FlatTensor
    private let audio: FlatTensor
    private let ref: FlatTensor
    private var lines: [String] = []

    public init(resources: URL) throws {
        self.resources = resources
        meta = try BundleMeta.load(from: resources)
        latents = try FlatTensor.load(meta.unet.latents, in: resources)
        audio = try FlatTensor.load(meta.unet.audio, in: resources)
        ref = try FlatTensor.load(meta.unet.ref, in: resources)
    }

    private func say(_ s: String) {
        lines.append(s)
        log(s)
    }

    private func modelURL(_ m: ModelMeta) -> URL { resources.appendingPathComponent(m.file) }

    /// Every UNet in the bundle: plain models, and chunk sets grouped by their parent (sorted by batch).
    public func unetSpecs() -> [UNetSpec] {
        var specs = meta.models(ofKind: "unet").map {
            UNetSpec(name: $0.name, batch: $0.batch, variant: $0.variant, layout: $0.layout, stages: [$0])
        }
        let chunks = meta.models(ofKind: "unet_chunk")
        for parent in Set(chunks.compactMap { $0.chunk_of }).sorted() {
            let stages = chunks.filter { $0.chunk_of == parent }.sorted { ($0.chunk_index ?? 0) < ($1.chunk_index ?? 0) }
            specs.append(UNetSpec(name: "\(parent)_chunked", batch: stages[0].batch, variant: stages[0].variant,
                                  layout: stages[0].layout, stages: stages))
        }
        return specs.sorted { ($0.batch, $0.name) < ($1.batch, $1.name) }
    }

    // MARK: - watchers

    private var pressureSource: DispatchSourceMemoryPressure? = nil
    private var thermalObserver: NSObjectProtocol? = nil

    /// Log memory-pressure and thermal-state changes as they happen (a kill usually follows a 'critical').
    private func startWatchers() {
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical, .normal], queue: .global(qos: .utility))
        src.setEventHandler { [weak self] in
            guard let self = self else { return }
            let ev = src.data
            let name = ev.contains(.critical) ? "CRITICAL" : ev.contains(.warning) ? "warning" : "normal"
            self.say("memory pressure \(name) (\(memLine()))")
        }
        src.resume()
        pressureSource = src
        thermalObserver = NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification,
                                                                 object: nil, queue: nil) { [weak self] _ in
            self?.say("thermal state -> \(thermalName(ProcessInfo.processInfo.thermalState))")
        }
        say("free disk \(freeDiskMB()) MB (\(memLine()))")
    }

    private var settleSeconds = 0.0
    private func settle() {
        guard settleSeconds > 0 else { return }
        say("settling \(settleSeconds) s (\(memLine()))")
        Thread.sleep(forTimeInterval: settleSeconds)
    }

    // MARK: - prediction

    /// The synchronous prediction (inside an async function Swift would otherwise pick the async overload).
    @inline(__always) private func predict(_ model: MLModel, _ p: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: p)
    }

    /// Run the stages back to back: each takes what it needs, by name, from the inputs and the earlier outputs.
    private func predictStages(_ models: [MLModel], _ p: MLFeatureProvider) throws -> MLFeatureProvider {
        if models.count == 1 { return try predict(models[0], p) }
        var feats: [String: MLFeatureValue] = [:]
        for n in p.featureNames { feats[n] = p.featureValue(for: n) }
        var last: MLFeatureProvider = p
        for (i, m) in models.enumerated() {
            var dict: [String: MLFeatureValue] = [:]
            for n in m.modelDescription.inputDescriptionsByName.keys {
                guard let v = feats[n] else { throw HarnessError.missingOutput("stage \(i + 1) input '\(n)'") }
                dict[n] = v
            }
            last = try predict(m, try MLDictionaryFeatureProvider(dictionary: dict))
            for n in last.featureNames { feats[n] = last.featureValue(for: n) }
        }
        return last
    }

    private func loadModel(_ m: ModelMeta, units: MLComputeUnits) throws -> MLModel {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        return try MLModel(contentsOf: modelURL(m), configuration: cfg)
    }

    /// Load every stage of a UNet; returns the models and the seconds each took.
    private func loadStages(_ spec: UNetSpec, units: MLComputeUnits) throws -> ([MLModel], [Double]) {
        var models: [MLModel] = [], secs: [Double] = []
        for stage in spec.stages {
            let t = now()
            models.append(try loadModel(stage, units: units))
            secs.append(now() - t)
        }
        return (models, secs)
    }

    // MARK: - run

    public func run(options: BenchOptions) async -> BenchReport {
        let iso = ISO8601DateFormatter()
        var report = BenchReport(started: iso.string(from: Date()), device: DeviceInfo.current(),
                                 bundle_created: meta.created, repo_commit: meta.repo_commit, gate: meta.gate,
                                 options: options.record)
        report.mac_reference = meta.mac_reference
        settleSeconds = options.settleSeconds
        startWatchers()
        say("device: \(report.device.machine) \(report.device.model) \(report.device.os) \(report.device.os_version), "
            + "\(report.device.memory_gb) GB, thermal \(report.device.thermal_state), runtime \(report.device.runtime)")

        var unets = options.unet ? unetSpecs() : []
        if !options.includeBatch4 { unets = unets.filter { $0.batch == 1 } }
        if let names = options.modelNames { unets = unets.filter { names.contains($0.name) } }
        say("UNets: \(unets.map { "\($0.name) (\($0.stages.count) stage\($0.stages.count == 1 ? "" : "s"))" }.joined(separator: ", "))")
        let accuracyFrames = min(options.accuracyFrames ?? meta.unet.frames, meta.unet.frames)

        for spec in unets {
            for (i, units) in options.unetUnits.enumerated() {
                if isCancelled() { break }
                report.unet_runs.append(await benchUNet(spec, units: units, frames: options.unetFrames,
                                                        accuracyFrames: accuracyFrames, plan: options.computePlan && i == 0,
                                                        reload: options.reload))
            }
            if options.gpuCheck && spec.batch == 1 && !isCancelled() {
                report.unet_runs.append(await benchUNet(spec, units: .cpuAndGPU, frames: 0,
                                                        accuracyFrames: min(options.gpuCheckFrames, meta.unet.frames),
                                                        plan: false, reload: false))
            }
        }

        if let b1 = unets.first(where: { $0.batch == 1 }) {
            if options.sustainedSeconds > 0 && !isCancelled() {
                report.sustained = sustained(b1, units: options.phaseUnits, seconds: options.sustainedSeconds)
                settle()
            }
            if options.chainFrames > 0, let taesd = meta.models(ofKind: "taesd").first(where: { $0.batch == 1 }) {
                for units in options.taesdUnits where !isCancelled() {
                    if let c = chain(b1, unetUnits: options.phaseUnits, taesd: taesd, taesdUnits: units, frames: options.chainFrames) {
                        report.chains.append(c)
                    }
                    settle()
                }
            }
        }
        if options.whisper && meta.whisper != nil && !isCancelled() {
            report.whisper = await whisper(plan: options.computePlan)
        }

        report.finished = iso.string(from: Date())
        report.log = lines
        say("done")
        return report
    }

    // MARK: - UNet

    /// Feature providers covering `frames` capture frames in groups of the model's batch (last group padded).
    private func unetProviders(_ spec: UNetSpec, frames: Int) throws -> [MLDictionaryFeatureProvider] {
        let ins = spec.inputs
        guard let sIn = ins.first(where: { $0.name == "sample" }) ?? ins.first,
              let aIn = ins.first(where: { $0.name == "audio" }) ?? ins.last else {
            throw HarnessError.noModel("\(spec.name) has no sample/audio inputs")
        }
        let sType = try mlDataType(sIn.dtype), aType = try mlDataType(aIn.dtype)
        let b = spec.batch
        var providers: [MLDictionaryFeatureProvider] = []
        var start = 0
        while start < frames {
            var idx = Array(start ..< min(start + b, frames))
            while idx.count < b { idx.append(idx.last!) }
            let sample = try makeArray(latents.frames(idx), shape: [b] + latents.frameShape, dtype: sType)
            let aud = try makeArray(audio.frames(idx), shape: [b] + audio.frameShape, dtype: aType)
            providers.append(try MLDictionaryFeatureProvider(dictionary: [sIn.name: sample, aIn.name: aud]))
            start += b
        }
        return providers
    }

    private func benchUNet(_ spec: UNetSpec, units: MLComputeUnits, frames: Int, accuracyFrames: Int, plan: Bool,
                           reload: Bool) async -> UNetRun {
        var run = UNetRun(model: spec.name, variant: spec.variant, layout: spec.layout, batch: spec.batch,
                          compute_units: unitsName(units))
        run.stages = spec.stages.count
        run.stage_files = spec.stages.map { $0.file }
        let tag = "\(spec.name) [\(run.compute_units)]"
        do {
            say("\(tag): loading \(spec.stages.count) stage(s) (\(memLine()))")
            var models: [MLModel]? = nil
            var (loaded, secs) = try loadStages(spec, units: units)
            models = loaded
            loaded = []
            run.load_first_s = round4(secs.reduce(0, +))
            run.stage_load_s = secs.map(round4)
            say("\(tag): loaded in \(String(format: "%.2f", run.load_first_s!)) s \(secs.map { String(format: "%.2f", $0) }) (\(memLine()))")
            if reload {
                models = nil  // release first: two copies of an 850M-parameter model would double the footprint
                (loaded, secs) = try loadStages(spec, units: units)
                models = loaded
                loaded = []
                run.load_cached_s = round4(secs.reduce(0, +))
                say("\(tag): reloaded in \(String(format: "%.2f", run.load_cached_s!)) s (\(memLine()))")
            }

            let providers = try unetProviders(spec, frames: accuracyFrames)
            let out = spec.outputName
            let perFrame = ref.frameElements

            func measure(_ mdls: [MLModel]) throws {
                // first prediction after load
                let t0 = now()
                _ = try predictStages(mdls, providers[0])
                run.first_predict_ms = round2((now() - t0) * 1000)
                say("\(tag): first prediction \(run.first_predict_ms!) ms (\(memLine()))")

                // gate pass over the capture set
                var preds = [Float](repeating: 0, count: accuracyFrames * perFrame)
                var frame = 0
                for p in providers {
                    try autoreleasepool {
                        let o = try output(predictStages(mdls, p), out)
                        let f = floats(o)
                        let take = min(spec.batch, accuracyFrames - frame)
                        preds.replaceSubrange(frame * perFrame ..< (frame + take) * perFrame, with: f[0 ..< take * perFrame])
                        frame += take
                    }
                }
                var files: [FileStats] = []
                for cf in meta.unet.files where cf.start + cf.count <= accuracyFrames {
                    let a = Array(preds[cf.start * perFrame ..< (cf.start + cf.count) * perFrame])
                    let r = ref.frames(Array(cf.start ..< cf.start + cf.count))
                    let s = diffStats(a, r)
                    files.append(FileStats(name: cf.name, frames: cf.count, avatar: cf.avatar, mae: s.mae, rmse: s.rmse,
                                           p95_abs: s.p95_abs, max_abs: s.max_abs,
                                           pass: s.mae <= meta.gate.mae && s.max_abs <= meta.gate.max_abs))
                }
                let acc = AccuracySummary(perFile: files, gate: meta.gate)
                run.accuracy = acc
                say("\(tag): gate mae mean \(String(format: "%.5f", acc.mae_mean)) / max \(String(format: "%.5f", acc.mae_max)); "
                    + "max_abs max \(String(format: "%.4f", acc.max_abs_max)); \(acc.frames) frames; \(acc.pass ? "PASS" : "FAIL")")

                // timing pass
                if frames > 0 {
                    let calls = Int((Double(frames) / Double(spec.batch)).rounded(.up))
                    var ms: [Double] = []
                    ms.reserveCapacity(calls)
                    for i in 0 ..< calls {
                        if isCancelled() { break }
                        let p = providers[i % providers.count]
                        try autoreleasepool {
                            let t = now()
                            _ = try predictStages(mdls, p)
                            ms.append((now() - t) * 1000)
                        }
                    }
                    let stats = TimingStats(ms: ms)
                    run.timing_per_call = stats
                    run.timing_per_frame = stats.perFrame(batch: spec.batch)
                    run.frames_timed = ms.count * spec.batch
                    say("\(tag): \(String(format: "%.2f", run.timing_per_frame!.mean_ms)) ms/frame mean, "
                        + "p99 \(String(format: "%.2f", run.timing_per_frame!.p99_ms)), max \(String(format: "%.2f", run.timing_per_frame!.max_ms)) "
                        + "over \(ms.count) calls of \(spec.batch) (\(memLine()))")
                }
            }
            try measure(models!)
            models = nil  // release before MLComputePlan loads its own copy
            run.thermal_after = thermalName(ProcessInfo.processInfo.thermalState)
            settle()

            if plan {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = units
                var merged = PlacementSummary()
                for stage in spec.stages {
                    if let p = await placement(url: modelURL(stage), configuration: cfg) {
                        for (k, v) in p.ops { merged.ops[k, default: 0] += v }
                        for (k, v) in p.cost_share { merged.cost_share[k, default: 0] += v / Double(spec.stages.count) }
                        merged.total_ops += p.total_ops
                        if let n = p.note { merged.note = (merged.note.map { $0 + "; " } ?? "") + n }
                    }
                    settle()
                }
                run.placement = merged
                say("\(tag): placement \(merged.ops) cost share \(merged.cost_share.mapValues { round2($0 * 100) })%")
            }
        } catch {
            run.error = "\(error)"
            say("\(tag): ERROR \(error) (\(memLine()))")
        }
        return run
    }

    // MARK: - compute plan

    private func placement(url: URL, configuration: MLModelConfiguration) async -> PlacementSummary? {
        guard #available(iOS 17.4, macOS 14.4, *) else {
            return PlacementSummary(note: "MLComputePlan needs iOS 17.4 / macOS 14.4")
        }
        do {
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
            guard case .program(let program) = plan.modelStructure, let main = program.functions["main"] else {
                return PlacementSummary(note: "not an ML Program")
            }
            var summary = PlacementSummary()
            var cost: [String: Double] = [:]
            func walk(_ block: MLModelStructure.Program.Block) {
                for op in block.operations {
                    if let usage = plan.deviceUsage(for: op) {
                        let name = deviceName(usage.preferred)
                        summary.ops[name, default: 0] += 1
                        summary.total_ops += 1
                        if let c = plan.estimatedCost(of: op) { cost[name, default: 0] += c.weight }
                    }
                    for b in op.blocks { walk(b) }
                }
            }
            walk(main.block)
            let total = cost.values.reduce(0, +)
            if total > 0 { summary.cost_share = cost.mapValues { round4($0 / total) } }
            return summary
        } catch {
            return PlacementSummary(note: "compute plan failed: \(error)")
        }
    }

    @available(iOS 17.4, macOS 14.4, *)
    private func deviceName(_ d: MLComputeDevice) -> String {
        switch d {
        case .cpu: return "CPU"
        case .gpu: return "GPU"
        case .neuralEngine: return "ANE"
        @unknown default: return "OTHER"
        }
    }

    // MARK: - sustained

    private func sustained(_ spec: UNetSpec, units: MLComputeUnits, seconds: Double) -> SustainedRun? {
        let tag = "\(spec.name) sustained \(Int(seconds)) s [\(unitsName(units))]"
        do {
            say("\(tag): starting (\(memLine()))")
            let (models, _) = try loadStages(spec, units: units)
            let providers = try unetProviders(spec, frames: meta.unet.frames)
            _ = try predictStages(models, providers[0])
            let thermalStart = thermalName(ProcessInfo.processInfo.thermalState)
            var stamps: [(t: Double, ms: Double)] = []
            let start = now()
            var i = 0
            while now() - start < seconds && !isCancelled() {
                try autoreleasepool {
                    let t = now()
                    _ = try predictStages(models, providers[i % providers.count])
                    stamps.append((t - start, (now() - t) * 1000))
                }
                i += 1
            }
            var buckets: [SustainedBucket] = []
            var b0 = 0.0
            while b0 < seconds {
                let b1 = min(b0 + 5, seconds)
                let inB = stamps.filter { $0.t >= b0 && $0.t < b1 }.map { $0.ms }
                if !inB.isEmpty {
                    let st = TimingStats(ms: inB)
                    buckets.append(SustainedBucket(t0_s: b0, t1_s: b1, frames: inB.count * spec.batch,
                                                   mean_ms: round2(st.mean_ms / Double(spec.batch)),
                                                   p99_ms: round2(st.p99_ms / Double(spec.batch)),
                                                   thermal: thermalName(ProcessInfo.processInfo.thermalState)))
                }
                b0 = b1
            }
            let overall = TimingStats(ms: stamps.map { $0.ms }).perFrame(batch: spec.batch)
            let run = SustainedRun(model: spec.name, compute_units: unitsName(units), seconds: seconds, total_frames: stamps.count * spec.batch,
                                   overall: overall, buckets: buckets, thermal_start: thermalStart,
                                   thermal_end: thermalName(ProcessInfo.processInfo.thermalState))
            say("\(tag): \(run.total_frames) frames, \(String(format: "%.2f", overall.mean_ms)) ms/frame mean; "
                + "first bucket \(buckets.first.map { "\($0.mean_ms)" } ?? "-") → last \(buckets.last.map { "\($0.mean_ms)" } ?? "-") ms; "
                + "thermal \(thermalStart) → \(run.thermal_end) (\(memLine()))")
            return run
        } catch {
            say("\(tag): ERROR \(error) (\(memLine()))")
            return nil
        }
    }

    // MARK: - UNet -> TAESD chain

    private func chain(_ spec: UNetSpec, unetUnits: MLComputeUnits, taesd: ModelMeta, taesdUnits: MLComputeUnits, frames: Int) -> ChainRun? {
        let tag = "chain \(spec.name) [\(unitsName(unetUnits))] → \(taesd.name) [\(unitsName(taesdUnits))]"
        do {
            say("\(tag): starting (\(memLine()))")
            let (models, _) = try loadStages(spec, units: unetUnits)
            let t = now()
            let dec = try loadModel(taesd, units: taesdUnits)
            let tDec = now() - t
            let providers = try unetProviders(spec, frames: meta.unet.frames)
            let latIn = taesd.inputs.first!
            let latType = try mlDataType(latIn.dtype)
            let facesName = taesd.outputs.first?.name ?? "faces"
            let out = spec.outputName
            // warm both models: the first prediction after load compiles for the chosen units (seconds)
            let warm = try output(predictStages(models, providers[0]), out)
            _ = try predict(dec, MLDictionaryFeatureProvider(dictionary: [latIn.name: try convert(warm, to: latType)]))
            var u: [Double] = [], d: [Double] = [], tot: [Double] = []
            var facesShape: [Int] = []
            for i in 0 ..< frames {
                if isCancelled() { break }
                try autoreleasepool {
                    let t0 = now()
                    let lat = try output(predictStages(models, providers[i % providers.count]), out)
                    let t1 = now()
                    let latArr = try convert(lat, to: latType)
                    let faces = try output(predict(dec, MLDictionaryFeatureProvider(dictionary: [latIn.name: latArr])), facesName)
                    let t2 = now()
                    if i == 0 { facesShape = faces.shape.map { $0.intValue } }
                    u.append((t1 - t0) * 1000); d.append((t2 - t1) * 1000); tot.append((t2 - t0) * 1000)
                }
            }
            let run = ChainRun(unet_model: spec.name, taesd_model: taesd.name, compute_units: unitsName(unetUnits),
                               taesd_units: unitsName(taesdUnits), frames: tot.count,
                               unet_ms: TimingStats(ms: u), taesd_ms: TimingStats(ms: d), total_ms: TimingStats(ms: tot),
                               taesd_load_s: round4(tDec))
            say("\(tag): \(String(format: "%.2f", run.total_ms.mean_ms)) ms/frame (UNet \(String(format: "%.2f", run.unet_ms.mean_ms)) + "
                + "TAESD \(String(format: "%.2f", run.taesd_ms.mean_ms))), faces \(facesShape) (\(memLine()))")
            return run
        } catch {
            say("\(tag): ERROR \(error) (\(memLine()))")
            return nil
        }
    }

    // MARK: - Whisper front end

    private func whisper(plan: Bool) async -> WhisperRun {
        var run = WhisperRun()
        guard let w = meta.whisper else { return run }
        do {
            let melRef = try FlatTensor.load(w.mel, in: resources)
            let featRef = try FlatTensor.load(w.features_ref, in: resources)
            var melArr: MLMultiArray? = nil

            if let lm = meta.models(ofKind: "whisper_logmel").first {
                let wave = try FlatTensor.load(w.waveform, in: resources)
                let model = try loadModel(lm, units: .cpuOnly)
                let inp = lm.inputs.first!
                let x = try makeArray(wave.frame(0), shape: inp.shape, dtype: try mlDataType(inp.dtype))
                let p = try MLDictionaryFeatureProvider(dictionary: [inp.name: x])
                var ms: [Double] = []
                var mel: MLMultiArray? = nil
                for i in 0 ..< 12 {
                    let t = now()
                    mel = try output(predict(model, p), lm.outputs.first?.name ?? "logmel")
                    if i >= 2 { ms.append((now() - t) * 1000) }
                }
                run.logmel_model = lm.name
                run.logmel_ms = TimingStats(ms: ms)
                let d = diffStats(floats(mel!), melRef.frame(0))
                run.logmel_vs_hf_max_abs = round4(d.max_abs)
                melArr = mel
                say("whisper log-mel [CPU_ONLY]: \(String(format: "%.2f", run.logmel_ms!.mean_ms)) ms / 30 s, max |diff| vs HF \(run.logmel_vs_hf_max_abs!)")
            }

            if let enc = meta.models(ofKind: "whisper_encoder").first {
                run.encoder_model = enc.name
                let inp = enc.inputs.first!
                let inType = try mlDataType(inp.dtype)
                let hfMel = try makeArray(melRef.frame(0), shape: inp.shape, dtype: inType)
                let p = try MLDictionaryFeatureProvider(dictionary: [inp.name: hfMel])
                let refFeat = featRef.frame(0)
                let outName = enc.outputs.first?.name ?? "features"
                for units in [MLComputeUnits.all, .cpuAndNeuralEngine] {
                    if isCancelled() { break }
                    let name = unitsName(units)
                    let t = now()
                    let model = try loadModel(enc, units: units)
                    run.encoder_load_s[name] = round4(now() - t)
                    var ms: [Double] = []
                    var feat: MLMultiArray? = nil
                    for i in 0 ..< 23 {
                        try autoreleasepool {
                            let t = now()
                            feat = try output(predict(model, p), outName)
                            if i >= 3 { ms.append((now() - t) * 1000) }
                        }
                    }
                    run.encoder_ms[name] = TimingStats(ms: ms)
                    let d = diffStats(floats(feat!), refFeat)
                    run.features_vs_fp32[name] = d
                    say("whisper encoder [\(name)]: \(String(format: "%.2f", ms.reduce(0, +) / Double(ms.count))) ms / 30 s; "
                        + "vs fp32 features max \(String(format: "%.3f", d.max_abs)) mean \(String(format: "%.5f", d.mae))")
                    if units == .all, let mel = melArr {
                        // one chained call: Core ML log-mel -> encoder
                        var tot: [Double] = []
                        for _ in 0 ..< 5 {
                            let t = now()
                            let m16 = try convert(mel, to: inType)
                            _ = try predict(model, MLDictionaryFeatureProvider(dictionary: [inp.name: m16]))
                            tot.append((now() - t) * 1000)
                        }
                        run.front_end_ms = round2(tot.reduce(0, +) / Double(tot.count) + (run.logmel_ms?.mean_ms ?? 0))
                    }
                }
                if plan {
                    let cfg = MLModelConfiguration()
                    cfg.computeUnits = .cpuAndNeuralEngine
                    run.placement = await placement(url: modelURL(enc), configuration: cfg)
                }
            }
        } catch {
            run.error = "\(error)"
            say("whisper: ERROR \(error)")
        }
        return run
    }
}
