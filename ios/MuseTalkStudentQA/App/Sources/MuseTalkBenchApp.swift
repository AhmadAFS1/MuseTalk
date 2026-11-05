import SwiftUI

@main
struct MuseTalkBenchApp: App {
    @StateObject private var model = BenchViewModel()
    @StateObject private var lipsync = LipsyncPlayer()
    @State private var tab = CommandLine.arguments.contains("--lipsync") ? 1 : 0

    var body: some Scene {
        WindowGroup {
            ContentView(model: model, lipsync: lipsync, tab: $tab)
                .onAppear {
                    // memory emulation first, so the models load with the headroom a smaller phone would give
                    if MemoryTrace.requested() { MemoryTrace.shared.start() }
                    MemoryBallast.applyFromLaunchArguments()
                    model.autorunIfRequested()
                    lipsync.autostartIfRequested()
                }
        }
    }
}
