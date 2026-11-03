import CoreML
import Foundation
import HarnessCore

/// Runs the harness on this Mac (Neural Engine included) against a Resources folder.
///
///   swift run -c release bench-mac --resources ios/MuseTalkBench/Resources --out docs/ios_unet_bench/harness_mac_m5pro.json
@main
struct BenchMac {
    static func usage() -> Never {
        print("""
        usage: bench-mac --resources <dir> [--out <file.json>] [--frames N] [--sustained S] [--chain N]
                         [--units ALL,CPU_AND_NE] [--no-gpu] [--no-b4] [--no-whisper] [--no-reload] [--accuracy-frames N]
                         [--models a,b] [--no-unet] [--taesd-units ALL,CPU_AND_GPU,CPU_ONLY]
                         [--lipsync-check <avatar dir> [--lipsync-frames N] [--lipsync-taesd-units GPU] [--defer-decode]] [--fast-prediction]
                         [--lipsync-session <avatar dir> --wav <file> [--preroll S]]
        """)
        exit(2)
    }

    /// Swift frame loop vs the Python references exported with the avatar, plus per-frame timing on this Mac.
    static func lipsyncCheck(resources: URL, avatarDir: URL, frames: Int, taesdUnits: MLComputeUnits? = nil, deferDecode: Bool = false) {
        do {
            let engine = try LipsyncEngine(resources: resources, avatarDir: avatarDir)
            engine.log = { print($0) }
            engine.taesdUnits = taesdUnits
            engine.deferDecode = deferDecode
            print("lipsync-check: UNet \(engine.spec.name)")
            try engine.load()
            try engine.prepareAudio()
            let w = try engine.checkWindows()
            print(String(format: "audio windows vs repo: max |diff| %.5f, mean %.6f over %d frames", w.maxAbs, w.meanAbs, w.frames))
            for r in try engine.checkFrames() {
                print(String(format: "frame %d vs Python composed: mean |diff| %.3f/255 (face box %.3f), max %d", r.frame, r.frameMeanAbs, r.faceMeanAbs, r.maxAbs))
            }
            var u: [Double] = [], d: [Double] = [], c: [Double] = []
            let n = min(frames, engine.frameCount)
            for i in 0 ..< n {
                let (_, t) = try engine.renderFrame(i)
                if i >= 3 { u.append(t.unetMs); d.append(t.decodeMs); c.append(t.composeMs) }
            }
            let mean = { (a: [Double]) in a.reduce(0, +) / Double(max(1, a.count)) }
            print(String(format: "frame loop over %d frames: UNet %.1f + TAESD %.1f + compose %.1f = %.1f ms/frame (sequential)",
                         n, mean(u), mean(d), mean(c), mean(u) + mean(d) + mean(c)))
            // the two-stage pipeline, free-running (no clock); wall time per frame should approach the Neural Engine stage alone
            let pipe = LipsyncPipeline(engine: engine, aheadFrames: Int.max)
            pipe.log = { print($0) }
            let sem = DispatchSemaphore(value: 0)
            let t0 = now()
            pipe.start(from: 0, count: n, emit: { _, img in _ = img.cgImage() }, done: { sem.signal() })
            sem.wait()
            let s = pipe.snapshot
            print(String(format: "pipelined over %d frames: %.1f ms/frame wall (ANE stage %.1f, CPU stage %.1f + CGImage, in parallel; TAESD %.1f in the %@ stage)",
                         s.composed, (now() - t0) * 1000 / Double(max(1, s.composed)), s.inferMs, s.composeMs, s.decodeMs,
                         deferDecode ? "compose" : "ANE"))
        } catch {
            FileHandle.standardError.write("lipsync-check: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    /// A whole spoken turn through `LipsyncSession` on this Mac: a WAV (any rate) → 16 kHz → windows → paced
    /// pipeline → frames polled at 60 Hz, silent (no audio device needed). Prints the same LIPSYNC_RESULT as the phone.
    static func lipsyncSession(resources: URL, avatarDir: URL, wav: URL, preroll: Double) {
        do {
            let engine = try LipsyncEngine(resources: resources, avatarDir: avatarDir)
            engine.log = { print($0) }
            try engine.load()
            let waveform = try loadWaveform16k(url: wav)
            print(String(format: "wav %@: %.2f s at 16 kHz (%d samples)", wav.lastPathComponent, Double(waveform.count) / 16000, waveform.count))
            let session = LipsyncSession(engine: engine)
            session.config.prerollSeconds = preroll
            session.config.playAudio = false
            session.log = { print($0) }
            try session.speak(waveform: waveform, playbackURL: nil)
            var frames = 0
            while session.state != .finished && session.state != .failed {
                if session.currentFrame() != nil { frames += 1 }
                usleep(16_000)
            }
            print("frames delivered to the display: \(frames)")
            if let r = session.lastResult { print("LIPSYNC_RESULT " + r.json) }
        } catch {
            FileHandle.standardError.write("lipsync-session: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }

    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        var resources: URL? = nil
        var out: URL? = nil
        var options = BenchOptions()
        var lipsyncAvatar: URL? = nil
        var lipsyncFrames = 48
        var lipsyncTaesdUnits: MLComputeUnits? = nil
        var deferDecode = false
        var sessionAvatar: URL? = nil
        var sessionWav: URL? = nil
        var sessionPreroll = 0.4
        func take() -> String { args.isEmpty ? { usage() }() : args.removeFirst() }
        while !args.isEmpty {
            let a = args.removeFirst()
            switch a {
            case "--resources": resources = URL(fileURLWithPath: take())
            case "--out": out = URL(fileURLWithPath: take())
            case "--frames": options.unetFrames = Int(take()) ?? options.unetFrames
            case "--sustained": options.sustainedSeconds = Double(take()) ?? options.sustainedSeconds
            case "--chain": options.chainFrames = Int(take()) ?? options.chainFrames
            case "--accuracy-frames": options.accuracyFrames = Int(take())
            case "--units": options.unetUnits = take().split(separator: ",").compactMap { unitsFromName(String($0)) }
            case "--no-gpu": options.gpuCheck = false
            case "--no-b4": options.includeBatch4 = false
            case "--no-whisper": options.whisper = false
            case "--no-plan": options.computePlan = false
            case "--no-reload": options.reload = false
            case "--models": options.modelNames = take().split(separator: ",").map(String.init)
            case "--no-unet": options.unet = false
            case "--settle": options.settleSeconds = Double(take()) ?? 0
            case "--phase-units": options.phaseUnits = unitsFromName(take()) ?? .all
            case "--lipsync-check": lipsyncAvatar = URL(fileURLWithPath: take())
            case "--lipsync-frames": lipsyncFrames = Int(take()) ?? lipsyncFrames
            case "--lipsync-taesd-units": lipsyncTaesdUnits = unitsFromName(take())
            case "--defer-decode": deferDecode = true
            case "--fast-prediction": ModelTuning.fastPrediction = true
            case "--unpack-tar":  // HarnessCore's tar reader on a pack_phone_bundle.py / pack_model_pack.py tar
                let tar = URL(fileURLWithPath: take()), dir = URL(fileURLWithPath: take())
                do { try TarArchive.extract(tar, to: dir); print("unpacked \(tar.lastPathComponent) into \(dir.path)") }
                catch { FileHandle.standardError.write("unpack: \(error)\n".data(using: .utf8)!); exit(1) }
                return
            case "--lipsync-session": sessionAvatar = URL(fileURLWithPath: take())
            case "--wav": sessionWav = URL(fileURLWithPath: take())
            case "--preroll": sessionPreroll = Double(take()) ?? sessionPreroll
            case "--taesd-units": options.taesdUnits = take().split(separator: ",").compactMap { unitsFromName(String($0)) }
            default: usage()
            }
        }
        guard let resources = resources else { usage() }
        if let avatarDir = sessionAvatar {
            guard let wav = sessionWav else { usage() }
            lipsyncSession(resources: resources, avatarDir: avatarDir, wav: wav, preroll: sessionPreroll)
            return
        }
        if let avatarDir = lipsyncAvatar {
            lipsyncCheck(resources: resources, avatarDir: avatarDir, frames: lipsyncFrames,
                         taesdUnits: lipsyncTaesdUnits, deferDecode: deferDecode)
            return
        }
        do {
            let harness = try Harness(resources: resources)
            harness.log = { line in
                FileHandle.standardError.write(("[\(String(format: "%.1f", now()))] " + line + "\n").data(using: .utf8)!)
            }
            let report = await harness.run(options: options)
            let data = try report.json()
            if let out = out {
                try FileManager.default.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: out)
                print("wrote \(out.path)")
            } else {
                print(String(decoding: data, as: UTF8.self))
            }
            print(report.summaryLines().joined(separator: "\n"))
        } catch {
            FileHandle.standardError.write("bench-mac: \(error)\n".data(using: .utf8)!)
            exit(1)
        }
    }
}
