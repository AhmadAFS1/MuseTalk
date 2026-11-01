import AVFoundation
import Combine
import CoreGraphics
import Foundation
import HarnessCore
import QuartzCore
import UIKit

/// Composed frames waiting for the display link, shared between the pipeline's CPU stage and the main thread.
/// Every method takes the (non-recursive) lock once and never calls another method while holding it: the first
/// phone run hung because a locked closure read `count`, which locks again.
final class FrameBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Int: CGImage] = [:]
    private var audioStartValue: Double? = nil   // monotonic seconds when the audio started
    private var startRequested = false
    private var doneValue = false

    func reset() { lock.lock(); frames = [:]; audioStartValue = nil; startRequested = false; doneValue = false; lock.unlock() }
    /// Stores a composed frame; true the first time the buffer holds `preroll` frames (the caller then starts the audio).
    func put(_ i: Int, _ img: CGImage, preroll: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        frames[i] = img
        if startRequested || frames.count < preroll { return false }
        startRequested = true
        return true
    }
    /// Marks the pipeline finished; true if the audio has not been started yet (a clip shorter than the pre-roll).
    func markDone() -> Bool {
        lock.lock(); defer { lock.unlock() }
        doneValue = true
        if startRequested { return false }
        startRequested = true
        return true
    }
    var count: Int { lock.lock(); defer { lock.unlock() }; return frames.count }
    var done: Bool { lock.lock(); defer { lock.unlock() }; return doneValue }
    var audioStart: Double? {
        get { lock.lock(); defer { lock.unlock() }; return audioStartValue }
        set { lock.lock(); audioStartValue = newValue; lock.unlock() }
    }
    /// The newest frame at or before `target`; drops everything older than it.
    func take(upTo target: Int) -> (Int, CGImage)? {
        lock.lock(); defer { lock.unlock() }
        guard let best = frames.keys.filter({ $0 <= target }).max(), let img = frames[best] else { return nil }
        for k in frames.keys where k < best { frames[k] = nil }
        return (best, img)
    }
}

@MainActor
final class LipsyncPlayer: NSObject, ObservableObject {
    enum State: String { case idle, loading, preroll, playing, finished, failed }

    @Published var state: State = .idle
    @Published var status = "Tap Play: loads the UNet chunks, TAESD and Whisper, then streams the clip."
    @Published var image: CGImage? = nil
    @Published var stats = ""
    @Published var avatarName = ""

    private var engine: LipsyncEngine? = nil
    private var pipeline: LipsyncPipeline? = nil
    private var player: AVAudioPlayer? = nil
    private var displayLink: CADisplayLink? = nil
    private let buffer = FrameBuffer()
    private var exitAfter = false
    private var shownIndex = -1
    private var shown = 0
    private var stalls = 0
    private var ticks = 0
    private var total = 0
    private var fps = 24

    var hasAvatar: Bool {
        guard let res = Bundle.main.url(forResource: "HarnessBundle", withExtension: nil) else { return false }
        return FileManager.default.fileExists(atPath: res.appendingPathComponent("avatar/avatar.json").path)
    }

    /// Launch arguments: --lipsync [--exit]
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
        status = "loading models (a first launch compiles for the Neural Engine, 30–60 s)…"
        image = nil
        UIApplication.shared.isIdleTimerDisabled = true
        let avatarDir = resources.appendingPathComponent("avatar")
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let engine = try LipsyncEngine(resources: resources, avatarDir: avatarDir)
                engine.log = { line in
                    print("[lipsync] \(line)")
                    Task { @MainActor [weak self] in self?.status = line }
                }
                try engine.load()
                try engine.prepareAudio()
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    self.engine = engine
                    self.avatarName = engine.avatar.meta.avatar_id
                    self.beginPlayback(engine)
                }
            } catch {
                print("[lipsync] ERROR \(error)")
                await MainActor.run { [weak self] in
                    self?.state = .failed
                    self?.status = "\(error)"
                    UIApplication.shared.isIdleTimerDisabled = false
                    if self?.exitAfter == true { exit(1) }
                }
            }
        }
    }

    private func beginPlayback(_ engine: LipsyncEngine) {
        fps = engine.avatar.fps
        total = engine.frameCount
        let url = engine.avatar.dir.appendingPathComponent(engine.avatar.meta.audio.playback)
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            player = try AVAudioPlayer(contentsOf: url)
            player?.prepareToPlay()
        } catch {
            print("[lipsync] audio player: \(error)")
        }
        buffer.reset()
        shownIndex = -1; shown = 0; stalls = 0; ticks = 0
        state = .preroll
        let preroll = min(total, fps)
        status = "pre-rolling \(preroll) frames…"
        let buf = buffer
        let fps = self.fps
        // Two stages on two threads: the Neural Engine (UNet chunks + TAESD) and the CPU (JPEG decode + blend + CGImage),
        // kept about one second ahead of the audio clock; frames the clock has passed are skipped, never rendered.
        let pipe = LipsyncPipeline(engine: engine, aheadFrames: fps)
        pipe.frameMs = 1000.0 / Double(fps)
        pipe.log = { print("[lipsync] \($0)") }
        pipe.targetIndex = { buf.audioStart.map { Int((now() - $0) * Double(fps)) } }
        pipe.downstreamCount = { buf.count }
        pipeline = pipe
        pipe.start(emit: { [weak self] i, img in
            guard let cg = img.cgImage() else { print("[lipsync] frame \(i): CGImage failed"); return }
            if buf.put(i, cg, preroll: preroll) { DispatchQueue.main.async { self?.startAudio() } }
        }, done: { [weak self] in
            // a clip shorter than the pre-roll: start anyway
            if buf.markDone() { DispatchQueue.main.async { self?.startAudio() } }
        })
        displayLink?.invalidate()
        let dl = CADisplayLink(target: self, selector: #selector(tick))
        dl.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        dl.add(to: .main, forMode: .common)
        displayLink = dl
    }

    private func startAudio() {
        guard state == .preroll else { return }
        let t0 = now()
        player?.play()
        buffer.audioStart = t0
        state = .playing
        status = "streaming \(avatarName) at \(fps) fps · \(total) frames"
        print("[lipsync] audio started after pre-roll; \(buffer.count) frames buffered")
    }

    @objc private func tick() {
        guard let start = buffer.audioStart else {
            if let (i, img) = buffer.take(upTo: 0), i != shownIndex { image = img; shownIndex = i }
            return
        }
        let t = now() - start
        let target = min(total - 1, Int(t * Double(fps)))
        if let (i, img) = buffer.take(upTo: target) {
            if i != shownIndex { image = img; shownIndex = i; shown += 1 }
        } else if target > shownIndex {
            stalls += 1
        }
        ticks += 1
        if ticks % 15 == 0 { updateStats(t: t) }
        let finished = buffer.done && (shownIndex >= total - 1 || t * Double(fps) > Double(total) + 0.5)
        if finished || t > Double(total) / Double(fps) + 2 { finish(t: t) }
    }

    private func updateStats(t: Double) {
        guard let s = pipeline?.snapshot else { return }
        let genFps = t > 0 ? Double(s.generated) / t : 0
        stats = String(format: "generated %d (%.1f fps) · ANE %.0f ms · CPU %.0f ms · lead %d · skipped %d · late %d · shown %d · stalls %d · buffer %d · t %.1f s",
                       s.generated, genFps, s.inferMs, s.composeMs, s.leadFrames, s.skipped, s.droppedLate, shown, stalls, buffer.count, t)
    }

    private func finish(t: Double) {
        displayLink?.invalidate(); displayLink = nil
        pipeline?.cancel()
        player?.stop()
        updateStats(t: t)
        state = .finished
        status = "finished"
        UIApplication.shared.isIdleTimerDisabled = false
        let s = pipeline?.snapshot ?? LipsyncPipeline.Stats()
        let result: [String: Any] = ["avatar": avatarName, "fps": fps, "frames": total, "seconds": t, "generated": s.generated,
                                     "generated_fps": t > 0 ? Double(s.generated) / t : 0,
                                     "gen_ms_per_frame": s.inferMs + s.composeMs, "stage_infer_ms": s.inferMs, "stage_compose_ms": s.composeMs,
                                     "skipped": s.skipped, "dropped_late": s.droppedLate, "lead_frames": s.leadFrames, "pipelined": true,
                                     "shown": shown, "stalls": stalls, "load_s": engine?.loadSeconds ?? 0,
                                     "audio_prep_s": engine?.audioSeconds ?? 0, "unet": engine?.spec.name ?? ""]
        if let data = try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]) {
            print("LIPSYNC_RESULT " + String(decoding: data, as: UTF8.self))
        }
        if exitAfter { exit(0) }
    }

    func stop() {
        pipeline?.cancel()
        finish(t: buffer.audioStart.map { now() - $0 } ?? 0)
    }
}
