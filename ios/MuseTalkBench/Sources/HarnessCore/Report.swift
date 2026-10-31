import CoreML
import Foundation

public func unitsName(_ u: MLComputeUnits) -> String {
    switch u {
    case .all: return "ALL"
    case .cpuAndNeuralEngine: return "CPU_AND_NE"
    case .cpuAndGPU: return "CPU_AND_GPU"
    case .cpuOnly: return "CPU_ONLY"
    @unknown default: return "UNKNOWN"
    }
}

public func unitsFromName(_ s: String) -> MLComputeUnits? {
    switch s.uppercased() {
    case "ALL": return .all
    case "CPU_AND_NE", "NE", "ANE": return .cpuAndNeuralEngine
    case "CPU_AND_GPU", "GPU": return .cpuAndGPU
    case "CPU_ONLY", "CPU": return .cpuOnly
    default: return nil
    }
}

/// Where the compute plan puts the ops (counts and estimated cost share per device).
public struct PlacementSummary: Codable {
    public var ops: [String: Int] = [:]
    public var cost_share: [String: Double] = [:]
    public var total_ops: Int = 0
    public var note: String? = nil
}

public struct FileStats: Codable {
    public let name: String
    public let frames: Int
    public let avatar: String?
    public let mae: Double
    public let rmse: Double
    public let p95_abs: Double
    public let max_abs: Double
    public let pass: Bool
}

/// Same summary as scripts/validate_unet_backend.py `summarize` + the gate verdict.
public struct AccuracySummary: Codable {
    public let frames: Int
    public let files: Int
    public let mae_mean: Double
    public let mae_max: Double
    public let max_abs_mean: Double
    public let max_abs_max: Double
    public let p95_abs_max: Double
    public let pass: Bool
    public let per_file: [FileStats]

    init(perFile: [FileStats], gate: GateSpec) {
        per_file = perFile
        files = perFile.count
        frames = perFile.reduce(0) { $0 + $1.frames }
        let maes = perFile.map { $0.mae }, maxes = perFile.map { $0.max_abs }
        mae_mean = maes.reduce(0, +) / Double(max(1, maes.count))
        mae_max = maes.max() ?? .nan
        max_abs_mean = maxes.reduce(0, +) / Double(max(1, maxes.count))
        max_abs_max = maxes.max() ?? .nan
        p95_abs_max = perFile.map { $0.p95_abs }.max() ?? .nan
        pass = perFile.allSatisfy { $0.pass }
    }
}

public struct UNetRun: Codable {
    public let model: String
    public let variant: String?
    public let layout: String?
    public let batch: Int
    public let compute_units: String
    public var stages: Int = 1
    public var stage_files: [String] = []
    public var stage_load_s: [Double]? = nil
    public var load_first_s: Double? = nil
    public var load_cached_s: Double? = nil
    public var first_predict_ms: Double? = nil
    public var accuracy: AccuracySummary? = nil
    public var timing_per_call: TimingStats? = nil
    public var timing_per_frame: TimingStats? = nil
    public var frames_timed: Int = 0
    public var placement: PlacementSummary? = nil
    public var thermal_after: String? = nil
    public var error: String? = nil
}

public struct SustainedBucket: Codable {
    public let t0_s: Double
    public let t1_s: Double
    public let frames: Int
    public let mean_ms: Double
    public let p99_ms: Double
    public let thermal: String
}

public struct SustainedRun: Codable {
    public let model: String
    public let compute_units: String
    public let seconds: Double
    public let total_frames: Int
    public let overall: TimingStats
    public let buckets: [SustainedBucket]
    public let thermal_start: String
    public let thermal_end: String
    public var error: String? = nil
}

public struct ChainRun: Codable {
    public let unet_model: String
    public let taesd_model: String
    public let compute_units: String
    public let taesd_units: String
    public let frames: Int
    public let unet_ms: TimingStats
    public let taesd_ms: TimingStats
    public let total_ms: TimingStats
    public let taesd_load_s: Double
    public var error: String? = nil
}

public struct WhisperRun: Codable {
    public var logmel_model: String? = nil
    public var logmel_ms: TimingStats? = nil
    public var logmel_vs_hf_max_abs: Double? = nil
    public var encoder_model: String? = nil
    public var encoder_load_s: [String: Double] = [:]
    public var encoder_ms: [String: TimingStats] = [:]
    public var features_vs_fp32: [String: DiffStats] = [:]
    public var front_end_ms: Double? = nil      // logmel (CPU) + encoder (ALL), one chained call
    public var placement: PlacementSummary? = nil
    public var error: String? = nil
}

public struct OptionsRecord: Codable {
    public let unet_frames: Int
    public let unet_units: [String]
    public let gpu_check: Bool
    public let gpu_check_frames: Int
    public let include_batch4: Bool
    public let sustained_seconds: Double
    public let chain_frames: Int
    public let whisper: Bool
    public let reload: Bool
    public let settle_seconds: Double
    public let compute_plan: Bool
    public let phase_units: String
}

public struct BenchReport: Codable {
    public var started: String
    public var finished: String? = nil
    public var harness_version: String = "1"
    public var device: DeviceInfo
    public var bundle_created: String?
    public var repo_commit: String?
    public var gate: GateSpec
    public var options: OptionsRecord
    public var unet_runs: [UNetRun] = []
    public var sustained: SustainedRun? = nil
    public var chains: [ChainRun] = []
    public var whisper: WhisperRun? = nil
    public var mac_reference: JSONValue? = nil
    public var log: [String] = []

    public func json() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        enc.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
        return try enc.encode(self)
    }

    /// A few lines a person can read on the phone.
    public func summaryLines() -> [String] {
        var lines: [String] = []
        lines.append("\(device.machine) \(device.os) \(device.os_version) · \(device.memory_gb) GB · \(device.runtime)")
        for r in unet_runs {
            if let e = r.error {
                lines.append("\(r.model) [\(r.compute_units)]: ERROR \(e)")
                continue
            }
            var s = "\(r.model) [\(r.compute_units)]"
            if let t = r.timing_per_frame { s += ": \(String(format: "%.1f", t.mean_ms)) ms/frame (p99 \(String(format: "%.1f", t.p99_ms)))" }
            if let a = r.accuracy {
                s += " · gate mae \(String(format: "%.4f", a.mae_mean)) max_abs \(String(format: "%.2f", a.max_abs_max)) \(a.pass ? "PASS" : "FAIL")"
            }
            if let l = r.load_first_s { s += " · load \(String(format: "%.1f", l)) s" }
            if let p = r.placement, p.total_ops > 0 {
                let ane = Double(p.ops["ANE"] ?? 0) / Double(p.total_ops) * 100
                s += " · \(String(format: "%.0f", ane))% ops on ANE"
            }
            lines.append(s)
        }
        if let s = sustained {
            let first = s.buckets.first?.mean_ms ?? .nan, last = s.buckets.last?.mean_ms ?? .nan
            lines.append("sustained \(String(format: "%.0f", s.seconds)) s: \(String(format: "%.1f", first)) → \(String(format: "%.1f", last)) ms/frame, thermal \(s.thermal_start) → \(s.thermal_end)")
        }
        for c in chains {
            lines.append("UNet+TAESD chain [TAESD \(c.taesd_units)]: \(String(format: "%.1f", c.total_ms.mean_ms)) ms/frame (UNet \(String(format: "%.1f", c.unet_ms.mean_ms)) + TAESD \(String(format: "%.1f", c.taesd_ms.mean_ms)))")
        }
        if let w = whisper {
            for (u, t) in w.encoder_ms.sorted(by: { $0.key < $1.key }) {
                var s = "Whisper encoder [\(u)]: \(String(format: "%.1f", t.mean_ms)) ms / 30 s"
                if let d = w.features_vs_fp32[u] { s += " · max err \(String(format: "%.2f", d.max_abs)) mean \(String(format: "%.4f", d.mae))" }
                lines.append(s)
            }
            if let l = w.logmel_ms { lines.append("log-mel: \(String(format: "%.1f", l.mean_ms)) ms / 30 s") }
        }
        return lines
    }
}
