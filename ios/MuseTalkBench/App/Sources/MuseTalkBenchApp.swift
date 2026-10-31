import SwiftUI

@main
struct MuseTalkBenchApp: App {
    @StateObject private var model = BenchViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .onAppear { model.autorunIfRequested() }
        }
    }
}
