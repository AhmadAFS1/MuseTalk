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
                         [--units ALL,CPU_AND_NE] [--no-gpu] [--no-b4] [--no-whisper] [--accuracy-frames N]
        """)
        exit(2)
    }

    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        var resources: URL? = nil
        var out: URL? = nil
        var options = BenchOptions()
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
            default: usage()
            }
        }
        guard let resources = resources else { usage() }
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
