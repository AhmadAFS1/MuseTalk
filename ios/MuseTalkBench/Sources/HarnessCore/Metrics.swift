import Foundation

/// Same statistics as scripts/validate_unet_backend.py `tensor_stats` (torch.quantile is linear interpolation).
public struct DiffStats: Codable {
    public let mae: Double
    public let rmse: Double
    public let p95_abs: Double
    public let max_abs: Double
}

public func diffStats(_ a: [Float], _ b: [Float]) -> DiffStats {
    precondition(a.count == b.count, "diffStats: \(a.count) vs \(b.count) elements")
    var diffs = [Double](repeating: 0, count: a.count)
    var sum = 0.0, sumSq = 0.0, maxAbs = 0.0
    for i in 0 ..< a.count {
        let d = abs(Double(a[i]) - Double(b[i]))
        diffs[i] = d
        sum += d
        sumSq += d * d
        if d > maxAbs { maxAbs = d }
    }
    diffs.sort()
    let n = Double(a.count)
    return DiffStats(mae: sum / n, rmse: (sumSq / n).squareRoot(), p95_abs: percentile(sorted: diffs, 0.95), max_abs: maxAbs)
}

/// torch.quantile default (linear): index = q * (n - 1).
public func percentile(sorted: [Double], _ q: Double) -> Double {
    guard !sorted.isEmpty else { return .nan }
    let pos = q * Double(sorted.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, sorted.count - 1)
    let frac = pos - Double(lo)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

public struct TimingStats: Codable {
    public let count: Int
    public let mean_ms: Double
    public let min_ms: Double
    public let p50_ms: Double
    public let p90_ms: Double
    public let p99_ms: Double
    public let max_ms: Double
    public let over_40ms: Int
    public let over_50ms: Int

    public init(ms: [Double]) {
        let s = ms.sorted()
        count = ms.count
        mean_ms = ms.isEmpty ? .nan : ms.reduce(0, +) / Double(ms.count)
        min_ms = s.first ?? .nan
        p50_ms = percentile(sorted: s, 0.5)
        p90_ms = percentile(sorted: s, 0.9)
        p99_ms = percentile(sorted: s, 0.99)
        max_ms = s.last ?? .nan
        over_40ms = ms.filter { $0 > 40 }.count
        over_50ms = ms.filter { $0 > 50 }.count
    }

    /// Timings of a batched prediction expressed per frame.
    public func perFrame(batch: Int) -> TimingStats {
        TimingStats(count: count, mean_ms: mean_ms / Double(batch), min_ms: min_ms / Double(batch),
                    p50_ms: p50_ms / Double(batch), p90_ms: p90_ms / Double(batch), p99_ms: p99_ms / Double(batch),
                    max_ms: max_ms / Double(batch), over_40ms: over_40ms, over_50ms: over_50ms)
    }

    init(count: Int, mean_ms: Double, min_ms: Double, p50_ms: Double, p90_ms: Double, p99_ms: Double, max_ms: Double,
         over_40ms: Int, over_50ms: Int) {
        self.count = count; self.mean_ms = mean_ms; self.min_ms = min_ms; self.p50_ms = p50_ms
        self.p90_ms = p90_ms; self.p99_ms = p99_ms; self.max_ms = max_ms
        self.over_40ms = over_40ms; self.over_50ms = over_50ms
    }
}

/// Monotonic seconds.
@inline(__always) public func now() -> Double {
    Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9
}

public func round4(_ x: Double) -> Double { (x * 1e4).rounded() / 1e4 }
public func round2(_ x: Double) -> Double { (x * 1e2).rounded() / 1e2 }
