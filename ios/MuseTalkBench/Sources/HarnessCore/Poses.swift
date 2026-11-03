import Foundation

// The worker's pose protocol on the phone's three caches (Lingua build plan §3.12, "design as built"): the slots a
// character has, the pose ids the client's plan speaks in, the compiled per-frame plan of a turn (the router's
// boundary snapping and forced switches), and the scheduler's crossfade curve and blend.

/// The three caches a phone has for a character; the worker's pose set names them the same way.
public enum PoseSlot: String, CaseIterable, Codable {
    case talking, idle, smiling
}

/// `pose_protocol.py`'s ids mapped as a three-cache pose set maps them: speaking_direct → talking, light_smile →
/// smiling, every other pose (neutral_resting, active_listening, nod_agree, empathetic_head_tilt) → idle.
public func poseSlot(forPoseId id: String) -> PoseSlot {
    switch id {
    case "speaking_direct": return .talking
    case "light_smile": return .smiling
    default: return .idle
    }
}

/// `reaction_intent` → the pose the worker queues on the idle track (`REACTION_POSE` in pose_protocol.py).
public func reactionPoseId(_ intent: String) -> String? {
    switch intent {
    case "acknowledge": return "nod_agree"
    case "warmth": return "light_smile"
    case "empathy": return "empathetic_head_tilt"
    default: return nil
    }
}

/// One cue of a `pose_plan` v2: switch to `poseId` at `atPermille` of the audio.
public struct PoseCue: Equatable {
    public let atPermille: Int
    public let poseId: String
    public init(atPermille: Int, poseId: String) { self.atPermille = atPermille; self.poseId = poseId }
}

/// The scheduler's crossfade curve (`hls_gpu_scheduler.py`): frame k of N, 0 = all anchor, 1 = all incoming.
public func crossfadeAlpha(_ k: Int, of n: Int) -> Double {
    0.5 - 0.5 * cos(Double.pi * Double(k + 1) / Double(n + 1))
}

/// Full-frame `anchor × (1 − alpha) + incoming × alpha`, fixed point; frames of different sizes are not blended.
public func crossfade(anchor: RGBAImage, incoming: RGBAImage, alpha: Double) -> RGBAImage {
    guard anchor.width == incoming.width, anchor.height == incoming.height, anchor.pixels.count == incoming.pixels.count else {
        return incoming
    }
    let a = UInt32(max(0, min(256, (alpha * 256).rounded())))
    if a >= 256 { return incoming }
    if a == 0 { return anchor }
    let ia = 256 - a
    var out = incoming
    out.pixels.withUnsafeMutableBufferPointer { o in
        anchor.pixels.withUnsafeBufferPointer { an in
            incoming.pixels.withUnsafeBufferPointer { inc in
                var p = 0
                let n = o.count
                while p < n {
                    o[p] = UInt8((UInt32(an[p]) * ia + UInt32(inc[p]) * a) >> 8)
                    p += 1
                }
            }
        }
    }
    return out
}

/// A turn's body-motion plan compiled onto frames: the motion-bank policy of the worker's three-cache characters
/// (`docs/MULTIPOSE_LTX_MUSETALK_IMPLEMENTATION.md` "Runtime algorithm", `three_pose_transition_analysis`), with
/// the bank's optical-flow bridge replaced by the scheduler's cosine crossfade over the same 0.3 s.
///
/// - A reply shorter than `shortReplySeconds`: every body frame is the idle clip, continuing its phase under the
///   lipsync.
/// - A longer reply: 0.3 s of idle, then a bridge into the talking clip at its frame 0; a smile cue (`light_smile`)
///   enters through the same bridge once the cooldown since the last switch has passed and never inside the
///   terminal window; the final return into idle starts 0.5 s before the end (a 0.3 s bridge plus ~0.2 s of idle
///   lipsync), landing on the phase the idle clip would have reached had it kept running.
/// - Clips play their original forward range and wrap (the bank never uses the reversed half); the idle phase is
///   never reset to zero.
public struct FramePlan {
    public struct Settings {
        /// `short_reply_seconds`: below it the whole reply stays on the idle body (the docs' pilot used 3; set to 10).
        public var shortReplySeconds = 10.0
        /// `bridge_seconds`: the blend between two bodies, and the lead of idle before the talking entry.
        public var bridgeSeconds = 0.3
        /// Idle lipsync after the terminal bridge, before the audio ends.
        public var terminalIdleSeconds = 0.2
        /// Minimum time after a switch before the next (plus the bridge).
        public var cooldownSeconds = 0.25
        /// Whether a reaction event may switch the idle track between turns (the six-pose path's behaviour; off under the bank policy).
        public var idleReactions = false
        public init() {}

        public func bridgeFrames(fps: Int) -> Int { max(2, Int((bridgeSeconds * Double(fps)).rounded(.up))) }
    }

    public struct Segment: Equatable {
        public let start: Int           // first live frame of the segment
        public let slot: PoseSlot
        public let poseId: String
        public let sourceOffset: Int    // cycle index of the clip at `start`
        public let crossfadeFrames: Int // frames blended from the previous pose's last frame, starting at `start`
    }

    /// What the pipeline needs for live frame `i`.
    public struct Entry {
        public let slot: PoseSlot
        public let cycle: Int
        public let segmentStart: Int
        /// (k, n): frame k of an n-frame crossfade, nil outside one.
        public let crossfade: (Int, Int)?
    }

    public let total: Int
    public let fps: Int
    public let segments: [Segment]
    /// Forward clip lengths per slot (the unique frames; clips wrap, never reverse).
    public let lengths: [PoseSlot: Int]
    /// The idle clip's phase at frame 0 of the turn.
    public let idlePhase: Int

    public var switches: Int { max(0, segments.count - 1) }
    public var shortReply: Bool { segments.count == 1 }

    /// `available` are the slots the engine really has; a missing slot falls back (smiling/idle → idle → talking).
    public static func resolve(_ slot: PoseSlot, available: Set<PoseSlot>) -> PoseSlot {
        if available.contains(slot) { return slot }
        if slot != .talking && available.contains(.idle) { return .idle }
        return .talking
    }

    /// `lengths` are the forward clip lengths (unique frames) per slot; `idlePhase` the idle track's position at the
    /// turn's first frame; `cues` the v2 plan's segments (its first cue names the initial speaking pose, which the
    /// bank policy replaces with its own idle lead-in and talking entry).
    public static func compile(total: Int, fps: Int, lengths: [PoseSlot: Int], available: Set<PoseSlot>, idlePhase: Int,
                               cues: [PoseCue], settings s: Settings = Settings()) -> FramePlan {
        let idle = resolve(.idle, available: available)
        let talking = resolve(.talking, available: available)
        let idleLen = max(1, lengths[idle] ?? 1)
        let phase0 = ((idlePhase % idleLen) + idleLen) % idleLen
        var segments = [Segment(start: 0, slot: idle, poseId: "neutral_resting", sourceOffset: phase0, crossfadeFrames: 0)]
        let plan = { FramePlan(total: total, fps: fps, segments: segments, lengths: lengths, idlePhase: phase0) }
        // a short reply: the idle body all the way, continuing its phase
        if Double(total) / Double(fps) < s.shortReplySeconds || talking == idle { return plan() }
        let bridge = s.bridgeFrames(fps: fps)
        let terminalStart = total - bridge - Int((s.terminalIdleSeconds * Double(fps)).rounded(.up))
        let cooldown = bridge + Int((s.cooldownSeconds * Double(fps)).rounded(.up))
        // the entry: 0.3 s of idle, then the bridge into the talking clip at its frame 0
        guard bridge + bridge < terminalStart else { return plan() }
        segments.append(Segment(start: bridge, slot: talking, poseId: "speaking_direct", sourceOffset: 0, crossfadeFrames: bridge))
        var lastSwitch = bridge
        // semantic cues (a smile, or a pose the bank plays on the idle body): after the cooldown, never into the terminal window
        for cue in cues where cue.atPermille > 0 {
            let slot = resolve(poseSlot(forPoseId: cue.poseId), available: available)
            guard let current = segments.last, slot != current.slot else { continue }
            var at = Int((Double(total) * Double(cue.atPermille) / 1000).rounded())
            if at < lastSwitch + cooldown { at = lastSwitch + cooldown }   // deferred, as the bank defers a late smile
            if at + bridge > terminalStart { continue }                     // refused: it would overlap the return
            let offset = slot == idle ? (phase0 + at) % idleLen : 0
            segments.append(Segment(start: at, slot: slot, poseId: cue.poseId, sourceOffset: offset, crossfadeFrames: bridge))
            lastSwitch = at
        }
        // the terminal return into idle, on the phase the idle clip would have reached
        if let last = segments.last, last.slot != idle, terminalStart > lastSwitch {
            segments.append(Segment(start: terminalStart, slot: idle, poseId: "neutral_resting",
                                    sourceOffset: (phase0 + terminalStart) % idleLen, crossfadeFrames: bridge))
        }
        return plan()
    }

    public func entry(_ i: Int) -> Entry {
        var seg = segments[0]
        for s in segments where s.start <= i { seg = s }
        let len = max(1, lengths[seg.slot] ?? 1)
        let k = i - seg.start
        return Entry(slot: seg.slot, cycle: (seg.sourceOffset + k) % len, segmentStart: seg.start,
                     crossfade: k < seg.crossfadeFrames ? (k, seg.crossfadeFrames) : nil)
    }

    /// Where the idle track continues after the turn: the phase the idle clip would have reached had it kept running.
    public var idleResumePhase: Int { (idlePhase + total) % max(1, lengths[FramePlan.resolve(.idle, available: Set(lengths.keys))] ?? 1) }
    public var lastSlot: PoseSlot { total > 0 ? entry(total - 1).slot : .talking }

    public var description: String {
        segments.map { "\($0.poseId)@\($0.start)(\($0.slot.rawValue), offset \($0.sourceOffset), fade \($0.crossfadeFrames))" }.joined(separator: " → ")
    }
}
