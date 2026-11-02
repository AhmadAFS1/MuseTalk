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

/// A frame for the display: a live (lipsynced) frame of the turn, or a frame of the idle track between turns.
public struct DisplayFrame {
    public let index: Int
    public let image: CGImage
    public let live: Bool
}

/// Decodes the idle track's next frames ahead of the display thread, so a 24 fps loop costs the display thread a
/// dictionary lookup instead of a JPEG decode. Bounded: keys more than two frames behind the last request go.
final class IdlePrefetch {
    private let lock = NSLock()
    private var cache: [String: (RGBAImage, CGImage)] = [:]
    private var inFlight: Set<String> = []
    private let queue = DispatchQueue(label: "lipsync.idle.prefetch", qos: .userInteractive)

    private func key(_ slot: PoseSlot, _ cycle: Int) -> String { "\(slot.rawValue)#\(cycle)" }

    func take(_ slot: PoseSlot, _ cycle: Int) -> (RGBAImage, CGImage)? {
        lock.lock(); defer { lock.unlock() }
        return cache[key(slot, cycle)]
    }

    func put(_ slot: PoseSlot, _ cycle: Int, _ value: (RGBAImage, CGImage)) {
        lock.lock(); cache[key(slot, cycle)] = value; lock.unlock()
    }

    /// Decode `cycles` of `slot` in the background when they are not cached; drop everything not in `keep`.
    func prefetch(_ slot: PoseSlot, cycles: [Int], keep: Set<Int>, decode: @escaping (Int) throws -> RGBAImage) {
        lock.lock()
        let keepKeys = Set(keep.map { key(slot, $0) })
        for k in cache.keys where !keepKeys.contains(k) { cache[k] = nil }
        let missing = cycles.filter { cache[key(slot, $0)] == nil && !inFlight.contains(key(slot, $0)) }
        for c in missing { inFlight.insert(key(slot, c)) }
        lock.unlock()
        for c in missing {
            queue.async { [weak self] in
                guard let self = self else { return }
                if let img = try? decode(c), let cg = img.cgImage() { self.put(slot, c, (img, cg)) }
                self.lock.lock(); self.inFlight.remove(self.key(slot, c)); self.lock.unlock()
            }
        }
    }

    func clear() { lock.lock(); cache = [:]; lock.unlock() }
}

/// One avatar on screen: the idle track between turns (the worker's pose protocol on the phone's caches), and
/// one spoken turn at a time: the windows for a waveform, the two-stage pipeline paced by an audio clock, a
/// pre-roll, and the frames the display asks for. Owns the audio player, which is the clock. Platform-neutral:
/// the iOS app drives `currentFrame()` from a `CADisplayLink`, the Mac bench from a loop. This is what the Lingua
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
        /// The router's drift limit and crossfade lengths (Poses.swift).
        public var poses = FramePlan.Settings()
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
        public var poseSwitches = 0, crossfaded = 0
        public var plan = ""
        public var generatedFps: Double { seconds > 0 ? Double(generated) / seconds : 0 }
        public var shownFps: Double { seconds > 0 ? Double(shown) / seconds : 0 }

        public var dictionary: [String: Any] {
            ["avatar": avatar, "unet": unet, "fps": fps, "frames": frames, "seconds": seconds,
             "generated": generated, "generated_fps": generatedFps, "shown": shown, "shown_fps": shownFps,
             "skipped": skipped, "dropped_late": droppedLate, "stalls": stalls, "lead_frames": leadFrames,
             "gen_ms_per_frame": stageInferMs + stageComposeMs, "stage_infer_ms": stageInferMs, "stage_compose_ms": stageComposeMs,
             "pipelined": true, "preroll_s": prerollSeconds, "preroll_frames": prerollFrames, "start_latency_s": startLatencySeconds,
             "audio_prep_s": audioPrepSeconds, "load_s": loadSeconds,
             "audio_played": audioPlayed, "audio_route": audioRoute, "system_volume": systemVolume,
             "pose_switches": poseSwitches, "crossfaded": crossfaded, "plan": plan]
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
    private var plan: FramePlan? = nil

    // the idle track (all under `lock`, read on the display thread)
    private var idleStart: Double? = nil          // monotonic seconds the idle clock started; nil = not running
    private var idlePhase0 = 0                    // cycle index at `idleStart`
    private var idleSlot: PoseSlot = .idle
    private var pendingIdleSlot: PoseSlot? = nil  // applied at the next loop boundary
    private var reactionShown = false             // one reaction pose per turn, as the worker does
    private var idleFrozenPhase: Int? = nil       // the worker's idle sync hold: held from speak() to the turn's end
    private var idleShownCycle = -1
    private var idleLastImage: RGBAImage? = nil   // what the idle track showed last (anchors for its crossfades)
    private var idleAnchor: RGBAImage? = nil
    private var idleSeamK = 0                     // frames already blended of the current idle-track crossfade
    private var idleSeamN = 0
    private var lastBoundaryCycle = -1
    private let prefetch = IdlePrefetch()

    public init(engine: LipsyncEngine, config: Config = Config()) {
        self.engine = engine
        self.config = config
        fps = engine.avatar.fps
        idleSlot = FramePlan.resolve(.idle, available: engine.availableSlots)
    }

    public var state: State { lock.lock(); defer { lock.unlock() }; return stateValue }
    private func setState(_ s: State) {
        lock.lock()
        let changed = stateValue != s
        stateValue = s
        lock.unlock()
        if changed { onStateChange?(s) }
    }

    /// The avatar's first frame, to show before anything is spoken (callers that do not run the idle track).
    public func restingFrame() throws -> CGImage? { try engine.avatar.frame(0).cgImage() }

    // MARK: - the idle track

    /// Starts the idle track at clip position `phase` of the idle cache: between turns `currentFrame()` plays it.
    public func startIdle(phase: Int = 0) {
        lock.lock()
        idleStart = now(); idlePhase0 = phase; idleFrozenPhase = nil; idleShownCycle = -1
        idleAnchor = nil; idleSeamK = 0; idleSeamN = 0; lastBoundaryCycle = -1
        lock.unlock()
    }

    public var idleRunning: Bool { lock.lock(); defer { lock.unlock() }; return idleStart != nil }

    /// The cycle index the idle track is at now (frozen while a turn runs).
    private func idleCycleNow() -> Int? {
        guard let start = idleStart else { return nil }
        if let frozen = idleFrozenPhase { return frozen }
        let cyc = max(1, engine.cycleLengths[idleSlot] ?? engine.bundle(idleSlot).meta.cycle_length)
        return (idlePhase0 + Int((now() - start) * Double(fps))) % cyc
    }

    /// A worker session event (`webrtc_manager.py`): a reaction queues its pose on the idle track at the next loop
    /// boundary, once per turn; the listening events and an abort put the track back on the idle cache.
    public func event(_ name: String, reactionIntent: String? = nil) {
        lock.lock(); defer { lock.unlock() }
        switch name {
        case "assistant_reaction_ready":
            guard let pose = reactionPoseId(reactionIntent ?? "none") else { return }
            let slot = FramePlan.resolve(poseSlot(forPoseId: pose), available: engine.availableSlots)
            if slot != idleSlot && !reactionShown { pendingIdleSlot = slot; reactionShown = true }
        case "assistant_turn_aborted", "user_speech_started", "user_speech_ended", "assistant_thinking":
            let neutral = FramePlan.resolve(.idle, available: engine.availableSlots)
            pendingIdleSlot = idleSlot == neutral ? nil : neutral
        default:
            break
        }
    }

    private func idleFrame() -> DisplayFrame? {
        lock.lock()
        guard var cycle = idleCycleNow() else { lock.unlock(); return nil }
        let frozen = idleFrozenPhase != nil
        // a pending pose switch lands on a loop boundary (every half cycle: the worker's MP4 wraps once per clip)
        let unique = max(1, engine.bundle(idleSlot).meta.unique_frames)
        if !frozen, let pending = pendingIdleSlot, cycle % unique == 0, cycle != lastBoundaryCycle {
            lastBoundaryCycle = cycle
            pendingIdleSlot = nil
            idleAnchor = idleLastImage
            idleSeamK = 0; idleSeamN = config.poses.boundaryCrossfade
            idleSlot = pending
            // keep the phase: the new clip continues from the same position, modulo its own cycle
            let cyc = max(1, engine.cycleLengths[idleSlot] ?? 1)
            idlePhase0 = cycle % cyc
            idleStart = now()
            cycle = idlePhase0
        }
        if cycle == idleShownCycle { lock.unlock(); return nil }
        let slot = idleSlot
        let anchor = idleAnchor, k = idleSeamK, n = idleSeamN
        lock.unlock()

        let decode: (Int) throws -> RGBAImage = { [engine] c in try engine.plainFrame(slot: slot, cycle: c) }
        let cyc = max(1, engine.cycleLengths[slot] ?? 1)
        var frame: (RGBAImage, CGImage)? = prefetch.take(slot, cycle)
        if frame == nil, let img = try? decode(cycle), let cg = img.cgImage() { frame = (img, cg) }
        if !frozen {
            prefetch.prefetch(slot, cycles: [(cycle + 1) % cyc, (cycle + 2) % cyc, (cycle + 3) % cyc],
                              keep: Set([cycle, (cycle + 1) % cyc, (cycle + 2) % cyc, (cycle + 3) % cyc]), decode: decode)
        }
        guard var (img, cg) = frame else { return nil }
        if let an = anchor, k < n {
            img = crossfade(anchor: an, incoming: img, alpha: crossfadeAlpha(k, of: n))
            guard let blended = img.cgImage() else { return nil }
            cg = blended
        }
        lock.lock()
        idleShownCycle = cycle
        idleLastImage = img
        if k < n { idleSeamK = k + 1 } else { idleAnchor = nil }
        lock.unlock()
        return DisplayFrame(index: cycle, image: cg, live: false)
    }

    // MARK: - a turn

    /// Speak `waveform` (16 kHz mono) on the talking cache, no pose plan.
    public func speak(waveform: [Float], playbackURL: URL?) throws { try speak(waveform: waveform, playbackURL: playbackURL, cues: []) }

    /// Speak `waveform` (16 kHz mono) with a `pose_plan` v2's cues: builds the windows (Whisper on the Neural
    /// Engine), compiles the plan onto the frames from the idle track's phase, pre-rolls, then starts the audio and
    /// the video together. Returns when the pipeline is running; progress arrives through the callbacks and
    /// `currentFrame()`. Call off the main thread: the window build takes 0.2–4 s.
    public func speak(waveform: [Float], playbackURL: URL?, cues: [PoseCue]) throws {
        let s = state
        guard s == .idle || s == .finished || s == .failed else { throw BundleError.busy("session is \(s.rawValue)") }
        speakAt = now()
        setState(.preparing)
        // the worker's idle sync hold: the idle track freezes at its phase, the first live segment starts there
        lock.lock()
        let phase = idleCycleNow() ?? 0
        if idleStart != nil { idleFrozenPhase = phase }
        let anchor = idleLastImage
        lock.unlock()
        do {
            try engine.prepareAudio(waveform: waveform)
        } catch {
            // the idle track must not stay frozen behind a turn that never started
            lock.lock(); idleFrozenPhase = nil; lock.unlock()
            setState(.failed)
            throw error
        }
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
                do { try session.setActive(true) } catch {
                    // a speech recognizer being torn down can still hold the session: one retry
                    usleep(100_000)
                    try session.setActive(true)
                }
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
        let compiled = FramePlan.compile(total: total, fps: fps, cycles: engine.cycleLengths, available: engine.availableSlots,
                                         idlePhase: phase, cues: cues, settings: config.poses)
        lock.lock(); plan = compiled; lock.unlock()
        log("poses: \(compiled.description) (idle phase \(phase), \(total) frames)")
        let pipe = LipsyncPipeline(engine: engine, aheadFrames: max(preroll, Int(config.aheadSeconds * Double(fps))))
        pipe.frameMs = 1000.0 / Double(fps)
        pipe.log = log
        pipe.plan = compiled
        pipe.seamAnchor = anchor
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

    /// Call at display rate: the frame to show now (nil = keep the last one). During a turn these are the lipsynced
    /// frames (stalls counted, the end detected); between turns the idle track's frames.
    public func currentFrame() -> DisplayFrame? {
        let s = state
        if s == .idle || s == .finished || s == .failed { return idleFrame() }
        guard let start = buffer.audioStart else {
            // during the pre-roll, show frame 0 of the turn as it lands (a crossfade from the frozen idle frame)
            if let (i, img) = buffer.take(upTo: 0), i != shownIndex { shownIndex = i; return DisplayFrame(index: i, image: img, live: true) }
            return nil
        }
        let t = now() - start
        let target = min(total - 1, Int(t * Double(fps)))
        var out: DisplayFrame? = nil
        if let (i, img) = buffer.take(upTo: target) {
            if i != shownIndex { shownIndex = i; shown += 1; out = DisplayFrame(index: i, image: img, live: true) }
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
        return String(format: "generated %d (%.1f fps) · ANE %.0f ms · CPU %.0f ms · lead %d · skipped %d · late %d · shown %d · stalls %d · buffer %d · fades %d · t %.1f s",
                      s.generated, t > 0 ? Double(s.generated) / t : 0, s.inferMs, s.composeMs, s.leadFrames, s.skipped, s.droppedLate,
                      shown, stalls, buffer.count, s.crossfaded, t)
    }

    /// Stop early (or after the end): cancels the pipeline, stops the audio, reports.
    public func stop() { finish(seconds: buffer.audioStart.map { now() - $0 } ?? 0) }

    /// The turn so far (the same fields as the final result), for a once-a-second stats report.
    public func snapshotResult() -> Result { buildResult(seconds: buffer.audioStart.map { now() - $0 } ?? 0) }

    private func finish(seconds: Double) {
        lock.lock()
        if finished { lock.unlock(); return }
        finished = true
        lock.unlock()
        pipeline?.cancel()
        player?.stop()
        let r = buildResult(seconds: seconds)
        lastResult = r
        // the idle track resumes one frame past the last live phase, on the idle cache, under a short crossfade
        // (the worker cuts to idle frame 0 here)
        lock.lock()
        if idleStart != nil {
            let neutral = FramePlan.resolve(.idle, available: engine.availableSlots)
            idleSlot = neutral
            pendingIdleSlot = nil
            let cyc = max(1, engine.cycleLengths[neutral] ?? 1)
            idlePhase0 = ((plan?.lastCycle ?? 0) + 1) % cyc
            idleStart = now()
            idleFrozenPhase = nil
            idleShownCycle = -1
            idleAnchor = pipeline?.lastEmitted
            idleSeamK = 0; idleSeamN = idleAnchor == nil ? 0 : config.poses.seamCrossfade
            lastBoundaryCycle = -1
        }
        reactionShown = false
        lock.unlock()
        prefetch.clear()
        setState(.finished)
        onFinished?(r)
    }

    private func buildResult(seconds: Double) -> Result {
        let s = pipeline?.snapshot ?? LipsyncPipeline.Stats()
        var r = Result()
        r.avatar = engine.avatar.meta.avatar_id; r.unet = engine.spec.name; r.fps = fps; r.frames = total; r.seconds = seconds
        r.generated = s.generated; r.shown = shown; r.skipped = s.skipped; r.droppedLate = s.droppedLate; r.stalls = stalls
        r.leadFrames = s.leadFrames; r.stageInferMs = s.inferMs; r.stageComposeMs = s.composeMs
        r.prerollSeconds = config.prerollSeconds; r.prerollFrames = preroll
        r.startLatencySeconds = (buffer.audioStart ?? now()) - speakAt
        r.audioPrepSeconds = engine.audioSeconds; r.loadSeconds = engine.loadSeconds
        r.audioPlayed = audioPlayed; r.audioRoute = audioRoute; r.systemVolume = systemVolume
        lock.lock()
        r.poseSwitches = plan?.switches ?? 0
        r.plan = plan?.description ?? ""
        lock.unlock()
        r.crossfaded = s.crossfaded
        return r
    }
}
