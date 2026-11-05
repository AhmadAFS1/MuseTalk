// swift-tools-version:5.9
// Isolated iPhone QA app using the shipping Lingua HarnessCore snapshot.
import PackageDescription

let package = Package(
    name: "MuseTalkStudentQA",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "HarnessCore", targets: ["HarnessCore"]),
    ],
    targets: [
        .target(name: "HarnessCore", path: "Sources/HarnessCore"),
    ],
    swiftLanguageVersions: [.v5]
)
