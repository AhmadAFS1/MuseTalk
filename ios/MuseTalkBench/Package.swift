// swift-tools-version:5.9
// MuseTalk on-device timing harness.
//
//   HarnessCore  - loads the compiled Core ML models and the gate captures from a HarnessBundle
//                  folder (built by scripts/ios/prepare_harness_bundle.py), times them, and
//                  scores the UNet against the server's fp16 outputs with the repo gate.
//   bench-mac    - command-line runner for this Mac's Neural Engine (sanity check of the
//                  harness against the Python bench numbers before it goes to a phone).
//   App/         - the iOS app (xcodegen project) that bundles HarnessBundle and runs the same code.
import PackageDescription

let package = Package(
    name: "MuseTalkBench",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "HarnessCore", targets: ["HarnessCore"]),
        .executable(name: "bench-mac", targets: ["bench-mac"]),
    ],
    targets: [
        .target(name: "HarnessCore", path: "Sources/HarnessCore"),
        .executableTarget(name: "bench-mac", dependencies: ["HarnessCore"], path: "Sources/bench-mac"),
    ],
    swiftLanguageVersions: [.v5]
)
