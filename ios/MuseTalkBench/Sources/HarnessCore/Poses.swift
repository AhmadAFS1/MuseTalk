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

/// A turn's pose plan compiled onto frames, the way `webrtc_pose_router.py` compiles a v2 plan once the audio is
/// decoded: `desired = round(T × at ÷ 1000)`; a cue within `driftSeconds` of the end is skipped; a cue within
/// `driftSeconds` of the current clip's next loop end snaps to that boundary (the clip keeps its phase, a short
/// crossfade); otherwise the switch is on the exact frame with a longer crossfade and the incoming clip restarts
/// at frame 0. The first segment starts at the idle track's phase (the worker's idle sync hold).
public struct FramePlan {
    public struct Settings {
        /// WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS
        public var driftSeconds = 0.75
        /// WEBRTC_POSE_CROSSFADE_FRAMES as the launcher sets it (a snapped switch, and idle-track switches)
        public var boundaryCrossfade = 2
        /// WEBRTC_POSE_FORCED_CROSSFADE_FRAMES (a switch on the exact frame)
        public var forcedCrossfade = 4
        /// idle → live and live → idle (the worker cuts here; the phone blends)
        public var seamCrossfade = 2
        public init() {}
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
    public let cycles: [PoseSlot: Int]

    public var switches: Int { max(0, segments.count - 1) }

    /// `available` are the slots the engine really has; a missing slot falls back (smiling/idle → idle → talking).
    public static func resolve(_ slot: PoseSlot, available: Set<PoseSlot>) -> PoseSlot {
        if available.contains(slot) { return slot }
        if slot != .talking && available.contains(.idle) { return .idle }
        return .talking
    }

    public static func compile(total: Int, fps: Int, cycles: [PoseSlot: Int], available: Set<PoseSlot>, idlePhase: Int,
                               cues: [PoseCue], settings s: Settings = Settings()) -> FramePlan {
        let drift = max(0, Int((Double(fps) * s.driftSeconds).rounded()))
        var segments: [Segment] = []
        let firstPose = cues.first?.poseId ?? "speaking_direct"
        let firstSlot = resolve(poseSlot(forPoseId: firstPose), available: available)
        let firstCycle = max(1, cycles[firstSlot] ?? 1)
        segments.append(Segment(start: 0, slot: firstSlot, poseId: firstPose, sourceOffset: ((idlePhase % firstCycle) + firstCycle) % firstCycle,
                                crossfadeFrames: s.seamCrossfade))
        for cue in cues.dropFirst() {
            guard let prev = segments.last else { break }
            let slot = resolve(poseSlot(forPoseId: cue.poseId), available: available)
            if slot == prev.slot { continue }  // no adjacent duplicates once mapped onto the caches
            let desired = Int((Double(total) * Double(cue.atPermille) / 1000).rounded())
            if desired >= total - drift || desired <= prev.start { continue }
            let cyc = max(1, cycles[prev.slot] ?? 1)
            let rel = prev.sourceOffset + (desired - prev.start)   // cycle position of the current clip at `desired`
            let down = desired - (rel % cyc)                          // its loop end at or before `desired`
            var best: Int? = nil
            for b in [down, down + cyc] where b > prev.start && b < total - drift && abs(b - desired) <= drift {
                if best == nil || abs(b - desired) < abs(best! - desired) { best = b }
            }
            if let b = best {
                segments.append(Segment(start: b, slot: slot, poseId: cue.poseId, sourceOffset: 0, crossfadeFrames: s.boundaryCrossfade))
            } else {
                segments.append(Segment(start: desired, slot: slot, poseId: cue.poseId, sourceOffset: 0, crossfadeFrames: s.forcedCrossfade))
            }
        }
        return FramePlan(total: total, fps: fps, segments: segments, cycles: cycles)
    }

    public func entry(_ i: Int) -> Entry {
        var seg = segments[0]
        for s in segments where s.start <= i { seg = s }
        let cyc = max(1, cycles[seg.slot] ?? 1)
        let k = i - seg.start
        return Entry(slot: seg.slot, cycle: (seg.sourceOffset + k) % cyc, segmentStart: seg.start,
                     crossfade: k < seg.crossfadeFrames ? (k, seg.crossfadeFrames) : nil)
    }

    /// The clip phase the last live frame leaves off at (the idle track resumes one past it).
    public var lastCycle: Int { total > 0 ? entry(total - 1).cycle : 0 }
    public var lastSlot: PoseSlot { total > 0 ? entry(total - 1).slot : .talking }

    public var description: String {
        segments.map { "\($0.poseId)@\($0.start)(\($0.slot.rawValue), offset \($0.sourceOffset), fade \($0.crossfadeFrames))" }.joined(separator: " → ")
    }
}
