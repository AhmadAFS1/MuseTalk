import Combine
import CoreML
import Foundation
import HarnessCore
import UIKit

/// Thread-safe cancel flag read by the harness from its worker thread.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func set() { lock.lock(); flag = true; lock.unlock() }
    func reset() { lock.lock(); flag = false; lock.unlock() }
}

@MainActor
final class BenchViewModel: ObservableObject {
    @Published var lines: [String] = []
    @Published var summary: [String] = []
    @Published var running = false
    @Published var finished = false
    @Published var reportURL: URL? = nil
    @Published var error: String? = nil

    // options
    @Published var frames = 200
    @Published var sustainedSeconds = 30.0
    @Published var includeBatch4 = true
    @Published var gpuCheck = true
    @Published var whisper = true

    let device = DeviceInfo.current()
    private(set) var meta: BundleMeta? = nil
    private(set) var resourcesURL: URL? = nil
    private let cancel = CancelFlag()
    private var exitAfterRun = false

    init() {
        if let url = Bundle.main.url(forResource: "HarnessBundle", withExtension: nil) {
            resourcesURL = url
            do { meta = try BundleMeta.load(from: url) } catch { self.error = "meta.json: \(error)" }
        } else {
            error = "HarnessBundle folder missing from the app bundle (run scripts/ios/prepare_harness_bundle.py, then rebuild)"
        }
    }

    var bundleLine: String {
        guard let m = meta else { return "no bundle" }
        let names = m.models.map { $0.name }.joined(separator: ", ")
        return "bundle \(m.created ?? "?") @ \(m.repo_commit ?? "?"): \(names); \(m.unet.frames) gate frames"
    }

    /// The Mac figures for the same model, from meta.json, for a side-by-side on the phone.
    var macReferenceLines: [String] {
        guard let ref = meta?.mac_reference, case .object(let gates)? = ref["gate"] else { return [] }
        return gates.sorted { $0.key < $1.key }.compactMap { name, g in
            guard let mae = g["mae_mean"]?.double, let mx = g["max_abs_max"]?.double else { return nil }
            return "Mac ANE \(name): gate mae \(String(format: "%.4f", mae)), max_abs \(String(format: "%.2f", mx))"
        }
    }

    func options() -> BenchOptions {
        var o = BenchOptions()
        o.unetFrames = frames
        o.sustainedSeconds = sustainedSeconds
        o.includeBatch4 = includeBatch4
        o.gpuCheck = gpuCheck
        o.whisper = whisper
        return o
    }

    /// Launch arguments (xcodebuild / devicectl): --autorun [--frames N] [--sustained S] [--no-b4] [--no-gpu] [--no-whisper] [--exit]
    func autorunIfRequested() {
        var args = Array(CommandLine.arguments.dropFirst())
        guard args.contains("--autorun"), !running, !finished else { return }
        while !args.isEmpty {
            let a = args.removeFirst()
            switch a {
            case "--frames": frames = Int(args.isEmpty ? "" : args.removeFirst()) ?? frames
            case "--sustained": sustainedSeconds = Double(args.isEmpty ? "" : args.removeFirst()) ?? sustainedSeconds
            case "--no-b4": includeBatch4 = false
            case "--no-gpu": gpuCheck = false
            case "--no-whisper": whisper = false
            case "--exit": exitAfterRun = true
            default: break
            }
        }
        start()
    }

    func start() {
        guard let url = resourcesURL, !running else { return }
        running = true
        finished = false
        lines = []
        summary = []
        reportURL = nil
        error = nil
        cancel.reset()
        UIApplication.shared.isIdleTimerDisabled = true
        let opts = options()
        let flag = cancel
        let exitAfter = exitAfterRun
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let harness = try Harness(resources: url)
                harness.isCancelled = { flag.isSet }
                harness.log = { line in
                    print("[bench] \(line)")  // progress on the devicectl / Xcode console
                    Task { @MainActor [weak self] in self?.lines.append(line) }
                }
                let report = await harness.run(options: opts)
                let data = try report.json()
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
                let machine = report.device.machine.replacingOccurrences(of: ",", with: "_")
                let out = docs.appendingPathComponent("bench_\(machine)_\(stamp).json")
                try data.write(to: out)
                // for `devicectl device process launch --console`
                print("BENCH_JSON_BEGIN")
                print(String(decoding: data, as: UTF8.self))
                print("BENCH_JSON_END")
                let summary = report.summaryLines()
                await MainActor.run { [weak self] in
                    self?.reportURL = out
                    self?.summary = summary
                    self?.running = false
                    self?.finished = true
                    UIApplication.shared.isIdleTimerDisabled = false
                }
                if exitAfter { exit(0) }
            } catch {
                await MainActor.run { [weak self] in
                    self?.error = "\(error)"
                    self?.running = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
                if exitAfter { exit(1) }
            }
        }
    }

    func stop() {
        cancel.set()
        lines.append("cancelling after the current phase…")
    }
}
