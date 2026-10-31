import Foundation

public struct DeviceInfo: Codable {
    public let machine: String        // hw.machine: "iPhone13,3" on the phone, "arm64" on a Mac
    public let model: String          // hw.model: "Mac16,..." on a Mac
    public let chip: String           // machdep.cpu.brand_string when available
    public let os: String
    public let os_version: String
    public let memory_gb: Double
    public let cpu_count: Int
    public let thermal_state: String
    public let low_power_mode: Bool
    public let runtime: String        // "iOS", "iOS app on Mac", "iOS Simulator", "macOS CLI"

    public static func current() -> DeviceInfo {
        let pi = ProcessInfo.processInfo
        var runtime = "macOS CLI"
        var os = "macOS"
        #if os(iOS)
        os = "iOS"
        #if targetEnvironment(simulator)
        runtime = "iOS Simulator"
        #else
        runtime = pi.isiOSAppOnMac ? "iOS app on Mac" : "iOS"
        #endif
        #endif
        return DeviceInfo(
            machine: sysctlString("hw.machine"),
            model: sysctlString("hw.model"),
            chip: sysctlString("machdep.cpu.brand_string"),
            os: os,
            os_version: pi.operatingSystemVersionString,
            memory_gb: round2(Double(pi.physicalMemory) / Double(1 << 30)),
            cpu_count: pi.processorCount,
            thermal_state: thermalName(pi.thermalState),
            low_power_mode: pi.isLowPowerModeEnabled,
            runtime: runtime)
    }
}

public func thermalName(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "" }
    return String(cString: buf)
}
