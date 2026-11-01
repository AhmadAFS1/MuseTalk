import AVFoundation
import CoreGraphics
import Foundation

/// Composed frames waiting for the display, shared between the pipeline's CPU stage and the display thread.
/// Every method takes the (non-recursive) lock once and never calls another method while holding it: the first
/// phone run hung because a locked closure read `count`, which locks again.
final class FrameBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var frames: [Int: CGImage] = [:]
    private var audioStartValue: Double? = nil   // monotonic seconds when the clock started
    private var startRequested = false
    private var doneValue = false

    func reset() { lock.lock(); frames = [:]; audioStartValue = nil; startRequested = false; doneValue = false; lock.unlock() }

    /// Stores a composed frame; true the first time the buffer holds `preroll` frames (the caller then starts the clock).
    func put(_ i: Int, _ img: CGImage, preroll: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        frames[i] = img
        if startRequested || frames.count < preroll { return false }
        startRequested = true
        return true
    }

    /// Marks the pipeline finished; true if the clock has not started yet (a clip shorter than the pre-roll).
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

/// One spoken turn on an avatar: the windows for a waveform, the two-stage pipeline paced by an audio clock, a
/// pre-roll, and the frames the display asks for. Owns the audio player, which is the clock. Platform-neutral: the
/// iOS app drives `currentFrame()` from a `CADisplayLink`, the Mac bench from a loop. This is what the Lingua
/// native module wraps.
public final class LipsyncSession {
    public struct Config {
        /// Frames buffered before the audio starts, in seconds of clip. With generation slower than the clip's
        /// frame rate the buffer never refills, so a long pre-roll only delays the start.
        public var prerollSeconds = 0.4
        /// How far ahead of the clock the pipeline may run (finished + in flight).
        public var aheadSeconds = 1.0
        /// Play `playbackURL` through AVAudioPlayer. Off, the clock is a free-running timer (Mac checks).
        public var playAudio = true
        public init() {}
    }

    public enum State: String { case idle, preparing, preroll, playing, finished, failed }

    /// Everything `LIPSYNC_RESULT` reports.
    public struct Result {
        public var avatar = "", unet = "", fps = 0, frames = 0, seconds = 0.0
        public var generated = 0, shown = 0, skipped = 0, droppedLate = 0, stalls = 0, leadFrames = 1
        public var stageInferMs = 0.0, stageComposeMs = 0.0
        public var prerollSeconds = 0.0, prerollFrames = 0, startLatencySeconds = 0.0  // speak() → clock start
        public var audioPrepSeconds = 0.0, loadSeconds = 0.0
        public var audioPlayed = false, audioRoute = "", systemVolume = 0.0
        public var generatedFps: Double { seconds > 0 ? Double(generated) / seconds : 0 }
        public var shownFps: Double { seconds > 0 ? Double(shown) / seconds : 0 }

        public var dictionary: [String: Any] {
            ["avatar": avatar, "unet": unet, "fps": fps, "frames": frames, "seconds": seconds,
             "generated": generated, "generated_fps": generatedFps, "shown": shown, "shown_fps": shownFps,
             "skipped": skipped, "dropped_late": droppedLate, "stalls": stalls, "lead_frames": leadFrames,
             "gen_ms_per_frame": stageInferMs + stageComposeMs, "stage_infer_ms": stageInferMs, "stage_compose_ms": stageComposeMs,
             "pipelined": true, "preroll_s": prerollSeconds, "preroll_frames": prerollFrames, "start_latency_s": startLatencySeconds,
             "audio_prep_s": audioPrepSeconds, "load_s": loadSeconds,
             "audio_played": audioPlayed, "audio_route": audioRoute, "system_volume": systemVolume]
        }

        public var json: String {
            guard let data = try? JSONSerialization.data(withJSONObject: dictionary, options: [.sortedKeys]) else { return "{}" }
            return String(decoding: data, as: UTF8.self)
        }
    }

    public let engine: LipsyncEngine
    public var config: Config
    public var log: (String) -> Void = { _ in }
    /// Called on whatever thread changed the state (the CPU stage, the display thread, the caller).
    public var onStateChange: ((State) -> Void)? = nil
    public var onClockStart: ((Double) -> Void)? = nil   // monotonic seconds; the audio starts here
    public var onFinished: ((Result) -> Void)? = nil
    public private(set) var lastResult: Result? = nil

    private let buffer = FrameBuffer()
    private let lock = NSLock()
    private var stateValue: State = .idle
    private var pipeline: LipsyncPipeline? = nil
    private var player: AVAudioPlayer? = nil
    private var total = 0
    private var fps = 24
    private var preroll = 0
    private var shownIndex = -1, shown = 0, stalls = 0
    private var speakAt = 0.0
    private var audioRoute = "", systemVolume = 0.0, audioPlayed = false
    private var finished = false

    public init(engine: LipsyncEngine, config: Config = Config()) {
        self.engine = engine
        self.config = config
        fps = engine.avatar.fps
    }

    public var state: State { lock.lock(); defer { lock.unlock() }; return stateValue }
    private func setState(_ s: State) {
        lock.lock()
        let changed = stateValue != s
        stateValue = s
        lock.unlock()
        if changed { onStateChange?(s) }
    }

    /// The avatar's first frame, to show before anything is spoken.
    public func restingFrame() throws -> CGImage? { try engine.avatar.frame(0).cgImage() }

    /// Speak `waveform` (16 kHz mono): builds the windows (Whisper on the Neural Engine), pre-rolls, then starts
    /// the audio and the video together. Returns when the pipeline is running; progress arrives through the
    /// callbacks and `currentFrame()`. Call off the main thread: the window build takes 0.2–4 s.
    public func speak(waveform: [Float], playbackURL: URL?) throws {
        let s = state
        guard s == .idle || s == .finished || s == .failed else { throw BundleError.busy("session is \(s.rawValue)") }
        speakAt = now()
        setState(.preparing)
        try engine.prepareAudio(waveform: waveform)
        total = engine.frameCount
        buffer.reset()
        shownIndex = -1; shown = 0; stalls = 0
        finished = false; audioPlayed = false
        preroll = max(1, min(total, Int((config.prerollSeconds * Double(fps)).rounded())))
        player = nil
        if config.playAudio, let url = playbackURL {
            do {
                #if os(iOS)
                let session = AVAudioSession.sharedInstance()
                try session.setCategory(.playback, mode: .default, options: [])  // .playback ignores the silent switch
                try session.setActive(true)
                audioRoute = session.currentRoute.outputs.map { $0.portName }.joined(separator: " + ")
                systemVolume = Double(session.outputVolume)
                #else
                audioRoute = "default"; systemVolume = 1
                #endif
                let p = try AVAudioPlayer(contentsOf: url)
                p.volume = 1
                p.prepareToPlay()
                player = p
                log(String(format: "audio: %@, %.1f s, route %@, system volume %.2f", url.lastPathComponent, p.duration, audioRoute, systemVolume))
                if systemVolume < 0.05 { log("audio: the device volume is at zero") }
            } catch {
                log("audio player: \(error)")
            }
        }
        let pipe = LipsyncPipeline(engine: engine, aheadFrames: max(preroll, Int(config.aheadSeconds * Double(fps))))
        pipe.frameMs = 1000.0 / Double(fps)
        pipe.log = log
        let buf = buffer, fps = self.fps, preroll = self.preroll
        pipe.targetIndex = { buf.audioStart.map { Int((now() - $0) * Double(fps)) } }
        pipe.downstreamCount = { buf.count }
        pipeline = pipe
        setState(.preroll)
        pipe.start(emit: { [weak self] i, img in
            guard let cg = img.cgImage() else { self?.log("frame \(i): CGImage failed"); return }
            if buf.put(i, cg, preroll: preroll) { self?.startClock() }
        }, done: { [weak self] in
            if buf.markDone() { self?.startClock() }  // a clip shorter than the pre-roll: start anyway
        })
    }

    private func startClock() {
        guard state == .preroll else { return }
        let t0 = now()
        audioPlayed = player?.play() ?? false
        buffer.audioStart = t0
        setState(.playing)
        log("clock started after pre-roll: play() \(audioPlayed), route \(audioRoute), system volume \(String(format: "%.2f", systemVolume)), \(buffer.count) frames buffered, \(String(format: "%.2f", t0 - speakAt)) s after speak()")
        onClockStart?(t0)
    }

    /// Call at display rate: the frame to show now (nil = keep the last one). Counts stalls and detects the end.
    public func currentFrame() -> (index: Int, image: CGImage)? {
        guard let start = buffer.audioStart else {
            // during the pre-roll, show frame 0 of the turn as it lands
            if let (i, img) = buffer.take(upTo: 0), i != shownIndex { shownIndex = i; return (i, img) }
            return nil
        }
        let t = now() - start
        let target = min(total - 1, Int(t * Double(fps)))
        var out: (Int, CGImage)? = nil
        if let (i, img) = buffer.take(upTo: target) {
            if i != shownIndex { shownIndex = i; shown += 1; out = (i, img) }
        } else if target > shownIndex {
            stalls += 1
        }
        let done = buffer.done && (shownIndex >= total - 1 || t * Double(fps) > Double(total) + 0.5)
        if done || t > Double(total) / Double(fps) + 2 { finish(seconds: t) }
        return out
    }

    /// The on-screen stats line.
    public var statsLine: String {
        let s = pipeline?.snapshot ?? LipsyncPipeline.Stats()
        let t = buffer.audioStart.map { now() - $0 } ?? 0
        return String(format: "generated %d (%.1f fps) · ANE %.0f ms · CPU %.0f ms · lead %d · skipped %d · late %d · shown %d · stalls %d · buffer %d · t %.1f s",
                      s.generated, t > 0 ? Double(s.generated) / t : 0, s.inferMs, s.composeMs, s.leadFrames, s.skipped, s.droppedLate,
                      shown, stalls, buffer.count, t)
    }

    /// Stop early (or after the end): cancels the pipeline, stops the audio, reports.
    public func stop() { finish(seconds: buffer.audioStart.map { now() - $0 } ?? 0) }

    private func finish(seconds: Double) {
        lock.lock()
        if finished { lock.unlock(); return }
        finished = true
        lock.unlock()
        pipeline?.cancel()
        player?.stop()
        let s = pipeline?.snapshot ?? LipsyncPipeline.Stats()
        var r = Result()
        r.avatar = engine.avatar.meta.avatar_id; r.unet = engine.spec.name; r.fps = fps; r.frames = total; r.seconds = seconds
        r.generated = s.generated; r.shown = shown; r.skipped = s.skipped; r.droppedLate = s.droppedLate; r.stalls = stalls
        r.leadFrames = s.leadFrames; r.stageInferMs = s.inferMs; r.stageComposeMs = s.composeMs
        r.prerollSeconds = config.prerollSeconds; r.prerollFrames = preroll
        r.startLatencySeconds = (buffer.audioStart ?? now()) - speakAt
        r.audioPrepSeconds = engine.audioSeconds; r.loadSeconds = engine.loadSeconds
        r.audioPlayed = audioPlayed; r.audioRoute = audioRoute; r.systemVolume = systemVolume
        lastResult = r
        setState(.finished)
        onFinished?(r)
    }
}
