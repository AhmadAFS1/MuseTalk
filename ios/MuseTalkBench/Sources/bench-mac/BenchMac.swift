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
                         [--lipsync-check <avatar dir> [--lipsync-frames N]]
        """)
        exit(2)
    }

    /// Swift frame loop vs the Python references exported with the avatar, plus per-frame timing on this Mac.
    static func lipsyncCheck(resources: URL, avatarDir: URL, frames: Int) {
        do {
            let engine = try LipsyncEngine(resources: resources, avatarDir: avatarDir)
            engine.log = { print($0) }
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
            print(String(format: "frame loop over %d frames: UNet %.1f + TAESD %.1f + compose %.1f = %.1f ms/frame",
                         n, mean(u), mean(d), mean(c), mean(u) + mean(d) + mean(c)))
        } catch {
            FileHandle.standardError.write("lipsync-check: \(error)\n".data(using: .utf8)!)
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
            case "--taesd-units": options.taesdUnits = take().split(separator: ",").compactMap { unitsFromName(String($0)) }
            default: usage()
            }
        }
        guard let resources = resources else { usage() }
        if let avatarDir = lipsyncAvatar {
            lipsyncCheck(resources: resources, avatarDir: avatarDir, frames: lipsyncFrames)
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
