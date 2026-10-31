import SwiftUI

struct ContentView: View {
    @ObservedObject var model: BenchViewModel

    var body: some View {
        NavigationStack {
            List {
                Section("Device") {
                    Text("\(model.device.machine) · \(model.device.os) \(model.device.os_version)")
                    Text("\(model.device.memory_gb, specifier: "%.1f") GB · \(model.device.cpu_count) cores · thermal \(model.device.thermal_state) · \(model.device.runtime)")
                        .font(.footnote).foregroundStyle(.secondary)
                    Text(model.bundleLine).font(.footnote).foregroundStyle(.secondary)
                    ForEach(model.macReferenceLines, id: \.self) { Text($0).font(.footnote).foregroundStyle(.secondary) }
                }

                Section("Options") {
                    Stepper("UNet timing frames: \(model.frames)", value: $model.frames, in: 50 ... 2000, step: 50)
                    Picker("Sustained run", selection: $model.sustainedSeconds) {
                        Text("off").tag(0.0)
                        Text("30 s").tag(30.0)
                        Text("60 s").tag(60.0)
                        Text("120 s").tag(120.0)
                    }
                    Toggle("Batch-4 UNet", isOn: $model.includeBatch4)
                    Toggle("GPU accuracy anchor (slow)", isOn: $model.gpuCheck)
                    Toggle("Whisper front end", isOn: $model.whisper)
                }
                .disabled(model.running)

                Section {
                    if model.running {
                        Button(role: .destructive) { model.stop() } label: {
                            Label("Stop after this phase", systemImage: "stop.circle")
                        }
                    } else {
                        Button { model.start() } label: {
                            Label("Run benchmark", systemImage: "play.circle.fill")
                        }
                        .disabled(model.resourcesURL == nil)
                    }
                    if let e = model.error { Text(e).foregroundStyle(.red).font(.footnote) }
                    Text("Keep the app in the foreground; the screen stays on while it runs.")
                        .font(.footnote).foregroundStyle(.secondary)
                }

                if !model.summary.isEmpty {
                    Section("Result") {
                        ForEach(model.summary, id: \.self) { Text($0).font(.footnote.monospaced()) }
                        if let url = model.reportURL {
                            ShareLink(item: url) { Label("Share JSON report", systemImage: "square.and.arrow.up") }
                            Text("Also in Files → On My iPhone → MuseTalk Bench").font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Log") {
                    ForEach(Array(model.lines.enumerated()), id: \.offset) { _, line in
                        Text(line).font(.caption.monospaced())
                    }
                }
            }
            .navigationTitle("MuseTalk Bench")
            .overlay(alignment: .bottom) {
                if model.running {
                    ProgressView().padding().background(.thinMaterial, in: Capsule()).padding()
                }
            }
        }
    }
}
