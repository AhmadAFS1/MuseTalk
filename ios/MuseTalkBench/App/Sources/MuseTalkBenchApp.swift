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
                    model.autorunIfRequested()
                    lipsync.autostartIfRequested()
                }
        }
    }
}
