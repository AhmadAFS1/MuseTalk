import Combine
import CoreGraphics
import Foundation
import HarnessCore
import QuartzCore
import UIKit

/// The Lipsync tab's view model: a thin UIKit wrapper around `LipsyncSession` (HarnessCore), which owns the
/// pipeline, the pre-roll, the audio clock and the stats. This class only loads the engine, drives a
/// `CADisplayLink`, publishes the frames and prints `LIPSYNC_RESULT`.
///
/// Launch arguments: --lipsync [--exit] [--avatar-url <tar url> [--avatar-sha256 <hex>] [--avatar-id <id>]]
///                   [--wav <file path or url>] [--preroll <seconds>] [--silent]
/// Without --avatar-url the avatar bundled in HarnessBundle/avatar is used; without --wav, its bundled clip.
@MainActor
final class LipsyncPlayer: NSObject, ObservableObject {
    enum State: String { case idle, loading, preroll, playing, finished, failed }

    struct LaunchOptions {
        var avatarURL: URL? = nil
        var avatarSHA256: String? = nil
        var avatarId: String? = nil
        var wav: String? = nil
        var preroll = 0.4
        var silent = false

        static func parse(_ args: [String]) -> LaunchOptions {
            var o = LaunchOptions()
            var i = 0
            func value() -> String? { i + 1 < args.count ? args[i + 1] : nil }
            while i < args.count {
                switch args[i] {
                case "--avatar-url": o.avatarURL = value().flatMap(URL.init(string:)); i += 1
                case "--avatar-sha256": o.avatarSHA256 = value(); i += 1
                case "--avatar-id": o.avatarId = value(); i += 1
                case "--wav": o.wav = value(); i += 1
                case "--preroll": o.preroll = value().flatMap(Double.init) ?? o.preroll; i += 1
                case "--silent": o.silent = true
                default: break
                }
                i += 1
            }
            return o
        }
    }

    @Published var state: State = .idle
    @Published var status = "Tap Play: loads the UNet chunks, TAESD and Whisper, then streams the clip."
    @Published var image: CGImage? = nil
    @Published var stats = ""
    @Published var avatarName = ""

    private var engine: LipsyncEngine? = nil
    private var session: LipsyncSession? = nil
    private var displayLink: CADisplayLink? = nil
    private var exitAfter = false
    private var ticks = 0
    private let options = LaunchOptions.parse(CommandLine.arguments)

    var hasAvatar: Bool {
        if options.avatarURL != nil { return true }
        guard let res = Bundle.main.url(forResource: "HarnessBundle", withExtension: nil) else { return false }
        return FileManager.default.fileExists(atPath: res.appendingPathComponent("avatar/avatar.json").path)
    }

    func autostartIfRequested() {
        guard CommandLine.arguments.contains("--lipsync"), state == .idle else { return }
        exitAfter = CommandLine.arguments.contains("--exit")
        start()
    }

    func start() {
        guard state == .idle || state == .finished || state == .failed else { return }
        guard let resources = Bundle.main.url(forResource: "HarnessBundle", withExtension: nil) else {
            state = .failed; status = "HarnessBundle missing"; return
        }
        state = .loading
        status = options.avatarURL != nil ? "downloading the avatar…" : "loading models (a first launch compiles for the Neural Engine, 30–60 s)…"
        image = nil
        stats = ""
        UIApplication.shared.isIdleTimerDisabled = true
        let opts = options
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let log: (String) -> Void = { line in
                    print("[lipsync] \(line)")
                    Task { @MainActor [weak self] in self?.status = line }
                }
                // the avatar: bundled, or a phone bundle downloaded into Application Support/avatars/<id>/
                let avatarDir: URL
                if let url = opts.avatarURL {
                    let root = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                        .appendingPathComponent("avatars")
                    let id = opts.avatarId ?? url.deletingPathExtension().lastPathComponent
                    avatarDir = try await RemoteBundle.fetchAvatarBundle(tarURL: url, sha256: opts.avatarSHA256, avatarId: id, root: root, log: log)
                } else {
                    avatarDir = resources.appendingPathComponent("avatar")
                }
                let engine = try LipsyncEngine(resources: resources, avatarDir: avatarDir)
                engine.log = log
                try engine.load()
                let resting = try engine.avatar.frame(0).cgImage()
                await MainActor.run { [weak self] in
                    self?.engine = engine
                    self?.avatarName = engine.avatar.meta.avatar_id
                    self?.image = resting
                }
                // the audio: --wav (a file path or a URL) or the bundled clip
                var playback = engine.playbackURL
                let waveform: [Float]
                if let w = opts.wav {
                    var local = URL(fileURLWithPath: w)
                    if let u = URL(string: w), let scheme = u.scheme, scheme.hasPrefix("http") {
                        local = FileManager.default.temporaryDirectory.appendingPathComponent("speak-" + (u.lastPathComponent.isEmpty ? "clip.wav" : u.lastPathComponent))
                        try await RemoteBundle.download(u, to: local, log: log)
                    }
                    waveform = try loadWaveform16k(url: local)
                    playback = local
                    log(String(format: "wav %@: %.1f s at 16 kHz", local.lastPathComponent, Double(waveform.count) / 16000))
                } else {
                    waveform = try engine.bundledWaveform()
                }
                let session = LipsyncSession(engine: engine)
                session.config.prerollSeconds = opts.preroll
                session.config.playAudio = !opts.silent
                session.log = log
                session.onStateChange = { [weak self] s in Task { @MainActor [weak self] in self?.apply(s) } }
                session.onFinished = { [weak self] r in Task { @MainActor [weak self] in self?.finish(r) } }
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    self.session = session
                    self.state = .preroll
                    self.status = "pre-rolling \(opts.preroll) s…"
                    self.startDisplayLink()
                }
                try session.speak(waveform: waveform, playbackURL: playback)
            } catch {
                print("[lipsync] ERROR \(error)")
                if MemoryTrace.shared.running { print(MemoryTrace.shared.resultLine()) }
                await MainActor.run { [weak self] in
                    self?.state = .failed
                    self?.status = "\(error)"
                    UIApplication.shared.isIdleTimerDisabled = false
                    if self?.exitAfter == true { exit(1) }
                }
            }
        }
    }

    private func startDisplayLink() {
        displayLink?.invalidate()
        ticks = 0
        let dl = CADisplayLink(target: self, selector: #selector(tick))
        dl.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        dl.add(to: .main, forMode: .common)
        displayLink = dl
    }

    private func apply(_ s: LipsyncSession.State) {
        switch s {
        case .playing:
            state = .playing
            let r = session?.lastResult
            status = "streaming \(avatarName) at \(engine?.avatar.fps ?? 0) fps · \(engine?.frameCount ?? 0) frames" + (r == nil ? "" : "")
        case .failed:
            state = .failed
        default:
            break
        }
    }

    @objc private func tick() {
        guard let s = session else { return }
        if let (_, img) = s.currentFrame() { image = img }
        ticks += 1
        if ticks % 15 == 0 { stats = s.statsLine }
    }

    private func finish(_ r: LipsyncSession.Result) {
        displayLink?.invalidate(); displayLink = nil
        stats = session?.statsLine ?? ""
        state = .finished
        status = String(format: "finished · %d shown of %d (%.1f fps) · audio %@ via %@, volume %.0f %%", r.shown, r.frames, r.shownFps,
                        r.audioPlayed ? "played" : "NOT played", r.audioRoute, r.systemVolume * 100)
        UIApplication.shared.isIdleTimerDisabled = false
        print("LIPSYNC_RESULT " + r.json)
        if MemoryTrace.shared.running { print(MemoryTrace.shared.resultLine()) }
        if exitAfter { exit(0) }
    }

    func stop() { session?.stop() }
}
