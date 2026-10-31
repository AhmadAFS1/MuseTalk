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

    public init() {}

    var record: OptionsRecord {
        OptionsRecord(unet_frames: unetFrames, unet_units: unetUnits.map(unitsName), gpu_check: gpuCheck,
                      gpu_check_frames: gpuCheckFrames, include_batch4: includeBatch4, sustained_seconds: sustainedSeconds,
                      chain_frames: chainFrames, whisper: whisper)
    }
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

    /// The synchronous prediction (inside an async function Swift would otherwise pick the async overload).
    @inline(__always) private func predict(_ model: MLModel, _ p: MLFeatureProvider) throws -> MLFeatureProvider {
        try model.prediction(from: p)
    }

    // MARK: - run

    public func run(options: BenchOptions) async -> BenchReport {
        let iso = ISO8601DateFormatter()
        var report = BenchReport(started: iso.string(from: Date()), device: DeviceInfo.current(),
                                 bundle_created: meta.created, repo_commit: meta.repo_commit, gate: meta.gate,
                                 options: options.record)
        report.mac_reference = meta.mac_reference
        say("device: \(report.device.machine) \(report.device.model) \(report.device.os) \(report.device.os_version), "
            + "\(report.device.memory_gb) GB, thermal \(report.device.thermal_state), runtime \(report.device.runtime)")

        var unets = meta.models(ofKind: "unet").sorted { $0.batch < $1.batch }
        if !options.includeBatch4 { unets = unets.filter { $0.batch == 1 } }
        let accuracyFrames = min(options.accuracyFrames ?? meta.unet.frames, meta.unet.frames)

        for m in unets {
            for units in options.unetUnits {
                if isCancelled() { break }
                report.unet_runs.append(await benchUNet(m, units: units, frames: options.unetFrames,
                                                        accuracyFrames: accuracyFrames, plan: options.computePlan))
            }
            if options.gpuCheck && m.batch == 1 && !isCancelled() {
                report.unet_runs.append(await benchUNet(m, units: .cpuAndGPU, frames: 0,
                                                        accuracyFrames: min(options.gpuCheckFrames, meta.unet.frames),
                                                        plan: false))
            }
        }

        if let b1 = unets.first(where: { $0.batch == 1 }) {
            if options.sustainedSeconds > 0 && !isCancelled() {
                report.sustained = sustained(b1, seconds: options.sustainedSeconds)
            }
            if options.chainFrames > 0 && !isCancelled(), let taesd = meta.models(ofKind: "taesd").first(where: { $0.batch == 1 }) {
                report.chain = chain(b1, taesd: taesd, frames: options.chainFrames)
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

    private func load(_ m: ModelMeta, units: MLComputeUnits) throws -> (MLModel, Double) {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        let t0 = now()
        let model = try MLModel(contentsOf: modelURL(m), configuration: cfg)
        return (model, now() - t0)
    }

    /// Feature providers covering `frames` capture frames in groups of the model's batch (last group padded).
    private func unetProviders(_ m: ModelMeta, frames: Int) throws -> [MLDictionaryFeatureProvider] {
        guard let sIn = m.input(named: "sample") ?? m.inputs.first, let aIn = m.input(named: "audio") ?? m.inputs.last else {
            throw HarnessError.noModel("\(m.name) has no sample/audio inputs")
        }
        let sType = try mlDataType(sIn.dtype), aType = try mlDataType(aIn.dtype)
        let b = m.batch
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

    private func outputName(_ m: ModelMeta) -> String { m.outputs.first?.name ?? "pred_latents" }

    private func benchUNet(_ m: ModelMeta, units: MLComputeUnits, frames: Int, accuracyFrames: Int, plan: Bool) async -> UNetRun {
        var run = UNetRun(model: m.name, variant: m.variant, layout: m.layout, batch: m.batch, compute_units: unitsName(units))
        let tag = "\(m.name) [\(run.compute_units)]"
        do {
            say("\(tag): loading")
            var (model, tLoad) = try load(m, units: units)
            run.load_first_s = round4(tLoad)
            say("\(tag): loaded in \(String(format: "%.2f", tLoad)) s; reloading for the cached figure")
            (model, tLoad) = try load(m, units: units)
            run.load_cached_s = round4(tLoad)

            let providers = try unetProviders(m, frames: accuracyFrames)
            let out = outputName(m)
            let perFrame = ref.frameElements

            // first prediction after load
            let t0 = now()
            _ = try predict(model,  providers[0])
            run.first_predict_ms = round2((now() - t0) * 1000)
            say("\(tag): first prediction \(run.first_predict_ms!) ms")

            // gate pass over the capture set
            var preds = [Float](repeating: 0, count: accuracyFrames * perFrame)
            var frame = 0
            for p in providers {
                try autoreleasepool {
                    let o = try output(predict(model,  p), out)
                    let f = floats(o)
                    let take = min(m.batch, accuracyFrames - frame)
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
                let calls = Int((Double(frames) / Double(m.batch)).rounded(.up))
                var ms: [Double] = []
                ms.reserveCapacity(calls)
                for i in 0 ..< calls {
                    if isCancelled() { break }
                    let p = providers[i % providers.count]
                    try autoreleasepool {
                        let t = now()
                        _ = try predict(model,  p)
                        ms.append((now() - t) * 1000)
                    }
                }
                let stats = TimingStats(ms: ms)
                run.timing_per_call = stats
                run.timing_per_frame = stats.perFrame(batch: m.batch)
                run.frames_timed = ms.count * m.batch
                say("\(tag): \(String(format: "%.2f", run.timing_per_frame!.mean_ms)) ms/frame mean, "
                    + "p99 \(String(format: "%.2f", run.timing_per_frame!.p99_ms)), max \(String(format: "%.2f", run.timing_per_frame!.max_ms)) "
                    + "over \(ms.count) calls of \(m.batch)")
            }
            run.thermal_after = thermalName(ProcessInfo.processInfo.thermalState)

            if plan {
                let cfg = MLModelConfiguration()
                cfg.computeUnits = units
                run.placement = await placement(url: modelURL(m), configuration: cfg)
                if let p = run.placement { say("\(tag): placement \(p.ops) cost share \(p.cost_share.mapValues { round2($0 * 100) })%") }
            }
        } catch {
            run.error = "\(error)"
            say("\(tag): ERROR \(error)")
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

    private func sustained(_ m: ModelMeta, seconds: Double) -> SustainedRun? {
        let tag = "\(m.name) sustained \(Int(seconds)) s [ALL]"
        do {
            say("\(tag): starting")
            let (model, _) = try load(m, units: .all)
            let providers = try unetProviders(m, frames: meta.unet.frames)
            _ = try predict(model,  providers[0])
            let thermalStart = thermalName(ProcessInfo.processInfo.thermalState)
            var stamps: [(t: Double, ms: Double)] = []
            let start = now()
            var i = 0
            while now() - start < seconds && !isCancelled() {
                try autoreleasepool {
                    let t = now()
                    _ = try predict(model,  providers[i % providers.count])
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
                    buckets.append(SustainedBucket(t0_s: b0, t1_s: b1, frames: inB.count * m.batch, mean_ms: round2(st.mean_ms / Double(m.batch)),
                                                   p99_ms: round2(st.p99_ms / Double(m.batch)), thermal: thermalName(ProcessInfo.processInfo.thermalState)))
                }
                b0 = b1
            }
            let overall = TimingStats(ms: stamps.map { $0.ms }).perFrame(batch: m.batch)
            let run = SustainedRun(model: m.name, compute_units: "ALL", seconds: seconds, total_frames: stamps.count * m.batch,
                                   overall: overall, buckets: buckets, thermal_start: thermalStart,
                                   thermal_end: thermalName(ProcessInfo.processInfo.thermalState))
            say("\(tag): \(run.total_frames) frames, \(String(format: "%.2f", overall.mean_ms)) ms/frame mean; "
                + "first bucket \(buckets.first.map { "\($0.mean_ms)" } ?? "-") → last \(buckets.last.map { "\($0.mean_ms)" } ?? "-") ms; "
                + "thermal \(thermalStart) → \(run.thermal_end)")
            return run
        } catch {
            say("\(tag): ERROR \(error)")
            return nil
        }
    }

    // MARK: - UNet -> TAESD chain

    private func chain(_ m: ModelMeta, taesd: ModelMeta, frames: Int) -> ChainRun? {
        let tag = "chain \(m.name) → \(taesd.name) [ALL]"
        do {
            say("\(tag): starting")
            let (unet, _) = try load(m, units: .all)
            let (dec, tDec) = try load(taesd, units: .all)
            let providers = try unetProviders(m, frames: meta.unet.frames)
            let latIn = taesd.inputs.first!
            let latType = try mlDataType(latIn.dtype)
            let facesName = taesd.outputs.first?.name ?? "faces"
            let out = outputName(m)
            // warm both models: the first prediction after load compiles for the chosen units (seconds)
            let warm = try output(predict(unet, providers[0]), out)
            _ = try predict(dec, MLDictionaryFeatureProvider(dictionary: [latIn.name: try convert(warm, to: latType)]))
            var u: [Double] = [], d: [Double] = [], tot: [Double] = []
            var facesShape: [Int] = []
            for i in 0 ..< frames {
                if isCancelled() { break }
                try autoreleasepool {
                    let t0 = now()
                    let lat = try output(predict(unet,  providers[i % providers.count]), out)
                    let t1 = now()
                    let latArr = try convert(lat, to: latType)
                    let faces = try output(predict(dec,  MLDictionaryFeatureProvider(dictionary: [latIn.name: latArr])), facesName)
                    let t2 = now()
                    if i == 0 { facesShape = faces.shape.map { $0.intValue } }
                    u.append((t1 - t0) * 1000); d.append((t2 - t1) * 1000); tot.append((t2 - t0) * 1000)
                }
            }
            let run = ChainRun(unet_model: m.name, taesd_model: taesd.name, compute_units: "ALL", frames: tot.count,
                               unet_ms: TimingStats(ms: u), taesd_ms: TimingStats(ms: d), total_ms: TimingStats(ms: tot),
                               taesd_load_s: round4(tDec))
            say("\(tag): \(String(format: "%.2f", run.total_ms.mean_ms)) ms/frame (UNet \(String(format: "%.2f", run.unet_ms.mean_ms)) + "
                + "TAESD \(String(format: "%.2f", run.taesd_ms.mean_ms))), faces \(facesShape)")
            return run
        } catch {
            say("\(tag): ERROR \(error)")
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
                let (model, _) = try load(lm, units: .cpuOnly)
                let inp = lm.inputs.first!
                let x = try makeArray(wave.frame(0), shape: inp.shape, dtype: try mlDataType(inp.dtype))
                let p = try MLDictionaryFeatureProvider(dictionary: [inp.name: x])
                var ms: [Double] = []
                var mel: MLMultiArray? = nil
                for i in 0 ..< 12 {
                    let t = now()
                    mel = try output(predict(model,  p), lm.outputs.first?.name ?? "logmel")
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
                    let (model, tLoad) = try load(enc, units: units)
                    run.encoder_load_s[name] = round4(tLoad)
                    var ms: [Double] = []
                    var feat: MLMultiArray? = nil
                    for i in 0 ..< 23 {
                        try autoreleasepool {
                            let t = now()
                            feat = try output(predict(model,  p), outName)
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
                            _ = try predict(model,  MLDictionaryFeatureProvider(dictionary: [inp.name: m16]))
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
