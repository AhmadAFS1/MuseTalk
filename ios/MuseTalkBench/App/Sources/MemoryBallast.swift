import Foundation
import HarnessCore

/// Emulates a phone with a smaller memory allowance on this one. A 4 GB iPhone (12, 12 mini, 13, 13 mini, SE 2022)
/// gives a foreground app about 2,098 MB before jetsam kills it; a 6 GB phone about 3,071 MB (measured here, §5.4
/// of docs/ios_coreml_implementation.md). Holding dirty, incompressible memory in the process before anything
/// loads leaves the models exactly the headroom the smaller phone would.
///
/// Launch arguments:
///   --ballast-mb N          hold N MB on top of everything else (e.g. a host app's own footprint)
///   --emulate-limit-mb N    hold (this process's limit − N) MB, so the headroom equals a phone whose limit is N MB
///   --mem-trace             sample the footprint every 250 ms, print moves of 64 MB or more, and a MEM_RESULT line
///                           at the end (implied by the two flags above)
enum MemoryBallast {
    private static var blocks: [UnsafeMutableRawPointer] = []
    private(set) static var heldMB = 0

    static func applyFromLaunchArguments(_ args: [String] = CommandLine.arguments) {
        var want = 0
        var i = 0
        while i < args.count {
            let value: Int? = i + 1 < args.count ? Int(args[i + 1]) : nil
            switch args[i] {
            case "--ballast-mb":
                if let v = value { want += max(0, v); i += 1 }
            case "--emulate-limit-mb":
                if let v = value {
                    i += 1
                    let footprint = memoryFootprintMB(), headroom = availableMemoryMB()
                    if headroom <= 0 {
                        print("[mem] --emulate-limit-mb: iOS reports no memory headroom here (not a phone?); ignored")
                    } else {
                        let limit = footprint + headroom
                        print(String(format: "[mem] this process may use %.0f MB here (footprint %.0f + headroom %.0f); emulating a %d MB limit",
                                     limit, footprint, headroom, v))
                        want += max(0, Int(limit.rounded()) - v)
                    }
                }
            default: break
            }
            i += 1
        }
        if want > 0 { hold(mb: want) }
    }

    /// 64 MB blocks, every 8 bytes written with a pseudo-random word: dirty, resident, and nothing for the memory
    /// compressor to gain, so the whole amount counts toward the footprint jetsam looks at.
    static func hold(mb: Int) {
        let t0 = CFAbsoluteTimeGetCurrent()
        var left = mb
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        while left > 0 {
            let blockMB = min(64, left)
            let bytes = blockMB << 20
            let block = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 16384)
            let words = block.bindMemory(to: UInt64.self, capacity: bytes / 8)
            for w in 0 ..< bytes / 8 {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                words[w] = seed
            }
            blocks.append(block)
            heldMB += blockMB
            left -= blockMB
        }
        print(String(format: "[mem] ballast %d MB held in %.1f s (%@)", heldMB, CFAbsoluteTimeGetCurrent() - t0, memLine()))
    }
}

/// Samples the process footprint and the headroom iOS still grants, on a utility queue, from app start to the end
/// of the run. Prints on the console (what `devicectl --console` shows) so the last line before a jetsam kill tells
/// how far the process got; `resultLine()` is one JSON line after LIPSYNC_RESULT / BENCH_JSON_END.
final class MemoryTrace: @unchecked Sendable {
    static let shared = MemoryTrace()

    private let lock = NSLock()
    private var timer: DispatchSourceTimer? = nil
    private var pressure: DispatchSourceMemoryPressure? = nil
    private var t0 = CFAbsoluteTimeGetCurrent()
    private var peakFootprint = 0.0
    private var minHeadroom = Double.infinity
    private var samples = 0
    private var lastPrinted = 0.0
    private var lastPrintTime = 0.0
    private var warnings = 0
    private var criticals = 0
    private(set) var running = false

    static func requested(_ args: [String] = CommandLine.arguments) -> Bool {
        args.contains("--mem-trace") || args.contains("--ballast-mb") || args.contains("--emulate-limit-mb")
    }

    func start(intervalMs: Int = 250) {
        lock.lock()
        guard !running else { lock.unlock(); return }
        running = true
        t0 = CFAbsoluteTimeGetCurrent()
        lock.unlock()
        let queue = DispatchQueue(label: "bench.memtrace", qos: .utility)
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(intervalMs))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
        self.timer = timer
        let src = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: queue)
        src.setEventHandler { [weak self] in
            guard let self = self else { return }
            let critical = src.data.contains(.critical)
            self.lock.lock()
            if critical { self.criticals += 1 } else { self.warnings += 1 }
            let t = CFAbsoluteTimeGetCurrent() - self.t0
            self.lock.unlock()
            print(String(format: "[mem] t=%.1f s memory pressure %@ (%@)", t, critical ? "CRITICAL" : "warning", memLine()))
        }
        src.resume()
        pressure = src
        print("[mem] trace started (\(memLine()), thermal \(thermalName(ProcessInfo.processInfo.thermalState)))")
    }

    private func sample() {
        let footprint = memoryFootprintMB(), headroom = availableMemoryMB(), now = CFAbsoluteTimeGetCurrent()
        var line: String? = nil
        lock.lock()
        samples += 1
        peakFootprint = max(peakFootprint, footprint)
        if headroom >= 0 { minHeadroom = min(minHeadroom, headroom) }
        if abs(footprint - lastPrinted) >= 64 || now - lastPrintTime >= 5 {
            lastPrinted = footprint
            lastPrintTime = now
            line = String(format: "[mem] t=%.1f s footprint %.0f MB, %.0f MB left (peak %.0f)", now - t0, footprint, headroom, peakFootprint)
        }
        lock.unlock()
        if let line = line { print(line) }
    }

    func resultLine() -> String {
        sample()
        lock.lock()
        let seconds = (CFAbsoluteTimeGetCurrent() - t0)
        let d: [String: Any] = [
            "ballast_mb": MemoryBallast.heldMB,
            "peak_footprint_mb": Int(peakFootprint.rounded()),
            "min_headroom_mb": minHeadroom.isFinite ? Int(minHeadroom.rounded()) : -1,
            "footprint_now_mb": Int(memoryFootprintMB().rounded()),
            "headroom_now_mb": Int(availableMemoryMB().rounded()),
            "pressure_warnings": warnings,
            "pressure_criticals": criticals,
            "samples": samples,
            "seconds": (seconds * 10).rounded() / 10,
            "thermal_at_end": thermalName(ProcessInfo.processInfo.thermalState),
        ]
        lock.unlock()
        let data = (try? JSONSerialization.data(withJSONObject: d, options: [.sortedKeys])) ?? Data()
        return "MEM_RESULT " + String(decoding: data, as: UTF8.self)
    }
}
