import CoreGraphics
import CoreML
import Foundation
import ImageIO

// MARK: - avatar bundle (scripts/ios/prepare_avatar_bundle.py)

public struct AvatarAudioMeta: Decodable {
    public let source: String
    public let seconds: Double
    public let sample_rate: Int
    public let samples: Int
    public let waveform: String
    public let playback: String
    public let windows_ref: TensorFile
    public let frames: Int
}

public struct AvatarCheckMeta: Decodable {
    public let frames: [Int]
}

public struct AvatarMeta: Decodable {
    public let avatar_id: String
    public let fps: Int
    public let video_fps: Int?
    public let frame_width: Int
    public let frame_height: Int
    public let unique_frames: Int
    public let cycle_length: Int
    public let ping_pong: Bool
    public let frames: [String]
    public let masks: [String]
    public let mask_width: Int
    public let mask_height: Int
    public let coords: [[Int]]
    public let mask_coords: [[Int]]
    public let latents: TensorFile
    /// The harness clip; absent in phone-only bundles (`prepare_avatar_bundle.py --phone-only`).
    public let audio: AvatarAudioMeta?
    public let check: AvatarCheckMeta?
}

/// 8-bit RGBA, row-major, no padding.
public struct RGBAImage {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width; self.height = height; self.pixels = pixels
    }

    public func cgImage() -> CGImage? {
        let data = Data(pixels)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

func decodeCGImage(_ url: URL) throws -> CGImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw HarnessError.badTensorSize("cannot decode \(url.lastPathComponent)")
    }
    return img
}

/// Decode to 8-bit RGBA (alpha ignored).
func decodeRGBA(_ url: URL) throws -> RGBAImage {
    let img = try decodeCGImage(url)
    let w = img.width, h = img.height
    var pixels = [UInt8](repeating: 0, count: w * h * 4)
    try pixels.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
            throw HarnessError.badTensorSize("no RGBA context for \(url.lastPathComponent)")
        }
        ctx.interpolationQuality = .none
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    return RGBAImage(width: w, height: h, pixels: pixels)
}

/// Decode to 8-bit gray.
func decodeGray(_ url: URL) throws -> (width: Int, height: Int, pixels: [UInt8]) {
    let img = try decodeCGImage(url)
    let w = img.width, h = img.height
    var pixels = [UInt8](repeating: 0, count: w * h)
    try pixels.withUnsafeMutableBytes { raw in
        guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
            throw HarnessError.badTensorSize("no gray context for \(url.lastPathComponent)")
        }
        ctx.interpolationQuality = .none
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    return (w, h, pixels)
}

// MARK: - blending (port of musetalk.utils.blending: prepare_image_blending_plan + fixed-point blend)

public struct BlendPlan {
    let clipX0: Int, clipY0: Int, clipX1: Int, clipY1: Int      // ROI in the frame
    let dstX0: Int, dstY0: Int, dstX1: Int, dstY1: Int          // where the face lands inside the ROI
    let srcX0: Int, srcY0: Int                                  // offset into the resized face
    public let faceW: Int, faceH: Int
    let alpha: [UInt8]                                          // ROI-sized mask (after shrink)
    var roiW: Int { clipX1 - clipX0 }
    var roiH: Int { clipY1 - clipY0 }
}

/// Same geometry as prepare_image_blending_plan with MUSETALK_BLEND_SHRINK_MASK_BBOX on.
public func makeBlendPlan(frameW: Int, frameH: Int, faceBox: [Int], cropBox: [Int],
                          mask: (width: Int, height: Int, pixels: [UInt8])) -> BlendPlan? {
    let (x, y, x1, y1) = (faceBox[0], faceBox[1], faceBox[2], faceBox[3])
    let (xs, ys, xe, ye) = (cropBox[0], cropBox[1], cropBox[2], cropBox[3])
    let faceW = max(0, x1 - x), faceH = max(0, y1 - y)
    guard faceW > 0, faceH > 0, xe > xs, ye > ys else { return nil }
    var clipX0 = max(0, xs), clipY0 = max(0, ys)
    var clipX1 = min(frameW, xe), clipY1 = min(frameH, ye)
    guard clipX0 < clipX1, clipY0 < clipY1 else { return nil }
    var maskX0 = clipX0 - xs, maskY0 = clipY0 - ys
    var maskX1 = maskX0 + (clipX1 - clipX0), maskY1 = maskY0 + (clipY1 - clipY0)
    let fullFaceX0 = x - xs, fullFaceY0 = y - ys
    let fullFaceX1 = fullFaceX0 + faceW, fullFaceY1 = fullFaceY0 + faceH

    // mask ROI, then shrink to its non-zero bounding box
    var roi: [UInt8] = []
    roi.reserveCapacity((maskX1 - maskX0) * (maskY1 - maskY0))
    for r in maskY0 ..< maskY1 {
        roi.append(contentsOf: mask.pixels[(r * mask.width + maskX0) ..< (r * mask.width + maskX1)])
    }
    var rw = maskX1 - maskX0, rh = maskY1 - maskY0
    var sx0 = rw, sx1 = -1, sy0 = rh, sy1 = -1
    for r in 0 ..< rh {
        for c in 0 ..< rw where roi[r * rw + c] != 0 {
            if c < sx0 { sx0 = c }
            if c > sx1 { sx1 = c }
            if r < sy0 { sy0 = r }
            if r > sy1 { sy1 = r }
        }
    }
    guard sx1 >= 0 else { return nil }
    sx1 += 1; sy1 += 1
    var shrunk: [UInt8] = []
    shrunk.reserveCapacity((sx1 - sx0) * (sy1 - sy0))
    for r in sy0 ..< sy1 { shrunk.append(contentsOf: roi[(r * rw + sx0) ..< (r * rw + sx1)]) }
    roi = shrunk
    clipX0 += sx0; clipX1 = clipX0 + (sx1 - sx0)
    clipY0 += sy0; clipY1 = clipY0 + (sy1 - sy0)
    maskX0 += sx0; maskX1 = maskX0 + (sx1 - sx0)
    maskY0 += sy0; maskY1 = maskY0 + (sy1 - sy0)
    rw = sx1 - sx0; rh = sy1 - sy0

    let placeX0 = max(maskX0, fullFaceX0), placeY0 = max(maskY0, fullFaceY0)
    let placeX1 = min(maskX1, fullFaceX1), placeY1 = min(maskY1, fullFaceY1)
    guard placeX0 < placeX1, placeY0 < placeY1 else { return nil }
    return BlendPlan(clipX0: clipX0, clipY0: clipY0, clipX1: clipX1, clipY1: clipY1,
                     dstX0: placeX0 - maskX0, dstY0: placeY0 - maskY0, dstX1: placeX1 - maskX0, dstY1: placeY1 - maskY0,
                     srcX0: placeX0 - fullFaceX0, srcY0: placeY0 - fullFaceY0, faceW: faceW, faceH: faceH, alpha: roi)
}

/// cv2.resize INTER_LINEAR on an RGB 8-bit image (src pixel centre = (dst + 0.5) * scale - 0.5, edges clamped).
public func resizeBilinearRGB(_ src: [UInt8], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: dstW * dstH * 3)
    let sx = Float(srcW) / Float(dstW), sy = Float(srcH) / Float(dstH)
    var xs0 = [Int](repeating: 0, count: dstW), xs1 = xs0, wx = [Float](repeating: 0, count: dstW)
    for dx in 0 ..< dstW {
        var fx = (Float(dx) + 0.5) * sx - 0.5
        if fx < 0 { fx = 0 }
        let x0 = min(Int(fx), srcW - 1)
        xs0[dx] = x0; xs1[dx] = min(x0 + 1, srcW - 1); wx[dx] = fx - Float(x0)
    }
    src.withUnsafeBufferPointer { s in
        out.withUnsafeMutableBufferPointer { o in
            for dy in 0 ..< dstH {
                var fy = (Float(dy) + 0.5) * sy - 0.5
                if fy < 0 { fy = 0 }
                let y0 = min(Int(fy), srcH - 1), y1 = min(y0 + 1, srcH - 1)
                let wy = fy - Float(y0)
                let r0 = y0 * srcW * 3, r1 = y1 * srcW * 3
                var di = dy * dstW * 3
                for dx in 0 ..< dstW {
                    let a = r0 + xs0[dx] * 3, b = r0 + xs1[dx] * 3, c = r1 + xs0[dx] * 3, d = r1 + xs1[dx] * 3
                    let w = wx[dx]
                    for ch in 0 ..< 3 {
                        let top = Float(s[a + ch]) * (1 - w) + Float(s[b + ch]) * w
                        let bot = Float(s[c + ch]) * (1 - w) + Float(s[d + ch]) * w
                        o[di + ch] = UInt8(max(0, min(255, (top * (1 - wy) + bot * wy).rounded())))
                    }
                    di += 3
                }
            }
        }
    }
    return out
}

/// get_image_blending_with_plan (fixed point): paste the resized face into the ROI through the mask.
public func compose(frame: RGBAImage, face256: [UInt8], plan: BlendPlan) -> RGBAImage {
    var out = frame
    let face = resizeBilinearRGB(face256, srcW: 256, srcH: 256, dstW: plan.faceW, dstH: plan.faceH)
    let rw = plan.roiW, rh = plan.roiH, fw = frame.width
    out.pixels.withUnsafeMutableBufferPointer { o in
        frame.pixels.withUnsafeBufferPointer { base in
            face.withUnsafeBufferPointer { f in
                plan.alpha.withUnsafeBufferPointer { al in
                    for r in 0 ..< rh {
                        let fy = plan.clipY0 + r
                        let inDstY = r >= plan.dstY0 && r < plan.dstY1
                        for c in 0 ..< rw {
                            let a = UInt32(al[r * rw + c])
                            if a == 0 { continue }
                            let fx = plan.clipX0 + c
                            let p = (fy * fw + fx) * 4
                            let inDst = inDstY && c >= plan.dstX0 && c < plan.dstX1
                            if inDst {
                                let sy = plan.srcY0 + (r - plan.dstY0), sx = plan.srcX0 + (c - plan.dstX0)
                                let fi = (sy * plan.faceW + sx) * 3
                                let ia = 255 - a
                                o[p] = UInt8((UInt32(f[fi]) * a + UInt32(base[p]) * ia) / 255)
                                o[p + 1] = UInt8((UInt32(f[fi + 1]) * a + UInt32(base[p + 1]) * ia) / 255)
                                o[p + 2] = UInt8((UInt32(f[fi + 2]) * a + UInt32(base[p + 2]) * ia) / 255)
                            }
                            // outside the pasted rectangle overlay == base: blend is the identity
                        }
                    }
                }
            }
        }
    }
    return out
}

// MARK: - avatar

public final class AvatarBundle {
    public let dir: URL
    public let meta: AvatarMeta
    public let latents: FlatTensor
    private var frameCache: [Int: RGBAImage] = [:]
    private var frameOrder: [Int] = []
    private var planCache: [Int: BlendPlan] = [:]
    private let lock = NSLock()

    public init(dir: URL) throws {
        self.dir = dir
        let data = try Data(contentsOf: dir.appendingPathComponent("avatar.json"))
        meta = try JSONDecoder().decode(AvatarMeta.self, from: data)
        latents = try FlatTensor.load(meta.latents, in: dir)
    }

    public var fps: Int { meta.fps }
    /// Frames of the bundled harness clip (0 for a phone-only bundle).
    public var totalFrames: Int { meta.audio?.frames ?? 0 }

    /// Cycle position -> unique frame index (ping-pong caches store the reversed half only once).
    public func uniqueIndex(_ i: Int) -> Int {
        let j = i % meta.cycle_length
        return (meta.ping_pong && j >= meta.unique_frames) ? meta.cycle_length - 1 - j : j
    }

    public func frame(_ u: Int) throws -> RGBAImage {
        lock.lock()
        if let f = frameCache[u] { lock.unlock(); return f }
        lock.unlock()
        let img = try decodeRGBA(dir.appendingPathComponent(meta.frames[u]))
        lock.lock()
        frameCache[u] = img
        frameOrder.append(u)
        if frameOrder.count > 24, let old = frameOrder.first { frameOrder.removeFirst(); frameCache[old] = nil }
        lock.unlock()
        return img
    }

    public func plan(_ u: Int) throws -> BlendPlan {
        lock.lock()
        if let p = planCache[u] { lock.unlock(); return p }
        lock.unlock()
        let mask = try decodeGray(dir.appendingPathComponent(meta.masks[u]))
        guard let p = makeBlendPlan(frameW: meta.frame_width, frameH: meta.frame_height, faceBox: meta.coords[u],
                                    cropBox: meta.mask_coords[u], mask: mask) else {
            throw HarnessError.badTensorSize("no blend plan for frame \(u)")
        }
        lock.lock(); planCache[u] = p; lock.unlock()
        return p
    }
}

// MARK: - audio windows (AudioProcessor.encode_whisper_feature + build_audio_prompts)

public struct AudioWindows {
    public let frames: Int
    public let fps: Int
    /// [frames, 50, 384] as Float32.
    public let data: [Float]

    public func window(_ i: Int) -> ArraySlice<Float> { data[(i * 50 * 384) ..< ((i + 1) * 50 * 384)] }
}

/// waveform (16 kHz, any length) -> per-frame UNet audio windows, through the Core ML log-mel and encoder. Audio
/// longer than one Whisper segment (30 s) is encoded segment by segment and the features concatenated before the
/// gather, as the repo's AudioProcessor does, so a window at a segment boundary sees both sides.
public func computeAudioWindows(waveform: [Float], fps: Int, logmel: MLModel, logmelMeta: ModelMeta,
                                encoder: MLModel, encoderMeta: ModelMeta) throws -> AudioWindows {
    let segSamples = 16000 * 30
    let layers = 5, dim = 384, steps = 10
    let lmIn = logmelMeta.inputs[0]
    let encIn = encoderMeta.inputs[0]
    var features: [Float] = []  // [actualTotal, 5, 384], 50 steps per second of audio
    var actualTotal = 0
    var offset = 0
    repeat {
        let n = max(0, min(waveform.count - offset, segSamples))
        var seg = [Float](repeating: 0, count: segSamples)
        if n > 0 { seg.replaceSubrange(0 ..< n, with: waveform[offset ..< offset + n]) }
        let wave = try makeArray(seg, shape: lmIn.shape, dtype: try mlDataType(lmIn.dtype))
        let mel = try output(logmel.prediction(from: MLDictionaryFeatureProvider(dictionary: [lmIn.name: wave])),
                             logmelMeta.outputs.first?.name ?? "logmel")
        let melIn = try convert(mel, to: try mlDataType(encIn.dtype))
        let feat = try output(encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [encIn.name: melIn])),
                              encoderMeta.outputs.first?.name ?? "features")
        let f = floats(feat)  // [1500, 5, 384]
        let actual = min(1500, Int((Double(n) / 16000.0 * 50.0).rounded(.down)))
        features.append(contentsOf: f[0 ..< actual * layers * dim])
        actualTotal += actual
        offset += segSamples
    } while offset < waveform.count
    let frames = Int((Double(waveform.count) / 16000.0 * Double(fps)).rounded(.down))
    let padL = 4, padR = 12
    let total = padL + actualTotal + padR
    var data = [Float](repeating: 0, count: frames * steps * layers * dim)
    data.withUnsafeMutableBufferPointer { out in
        features.withUnsafeBufferPointer { f in
            for i in 0 ..< frames {
                let a = (i * 50) / fps
                for s in 0 ..< steps {
                    let t = a + s  // index into the padded timeline
                    let src = t - padL
                    guard t < total, src >= 0, src < actualTotal else { continue }  // zero padding
                    for l in 0 ..< layers {
                        let o = ((i * steps + s) * layers + l) * dim
                        let si = (src * layers + l) * dim
                        for d in 0 ..< dim { out[o + d] = f[si + d] }
                    }
                }
            }
        }
    }
    return AudioWindows(frames: frames, fps: fps, data: data)
}

// MARK: - the frame loop

public struct FrameTiming {
    public let unetMs: Double
    public let decodeMs: Double
    public let composeMs: Double
    public var totalMs: Double { unetMs + decodeMs + composeMs }
}

/// Everything one talking-head frame needs, loaded once.
public final class LipsyncEngine {
    public let harness: Harness
    /// The talking cache: what a single-avatar caller (the harness app, the gate) lipsyncs on.
    public let avatar: AvatarBundle
    /// Every cache the session can draw or lipsync on, by slot (Poses.swift); `.talking` is always present.
    public let avatars: [PoseSlot: AvatarBundle]
    public let spec: UNetSpec
    public let units: MLComputeUnits
    public var log: (String) -> Void = { print($0) }

    private var unet: StagedUNet? = nil
    private var taesd: MLModel? = nil
    private var taesdMeta: ModelMeta? = nil
    private var sampleArrays: [PoseSlot: [MLMultiArray]] = [:]
    private var sampleName = "sample", audioName = "audio"
    private var audioDType: MLMultiArrayDataType = .float16
    public private(set) var windows: AudioWindows? = nil
    public private(set) var loadSeconds = 0.0
    public private(set) var audioSeconds = 0.0

    public convenience init(resources: URL, avatarDir: URL, units: MLComputeUnits = .cpuAndNeuralEngine, preferChunked: Bool = true) throws {
        try self.init(resources: resources, avatarDirs: [.talking: avatarDir], units: units, preferChunked: preferChunked)
    }

    /// `avatarDirs` must hold `.talking`; `.idle` and `.smiling` are what the character has on this phone.
    public init(resources: URL, avatarDirs: [PoseSlot: URL], units: MLComputeUnits = .cpuAndNeuralEngine, preferChunked: Bool = true) throws {
        harness = try Harness(resources: resources)
        var loaded: [PoseSlot: AvatarBundle] = [:]
        for (slot, dir) in avatarDirs { loaded[slot] = try AvatarBundle(dir: dir) }
        guard let talking = loaded[.talking] else { throw HarnessError.noModel("no talking avatar bundle") }
        avatars = loaded
        avatar = talking
        self.units = units
        let specs = harness.unetSpecs().filter { $0.batch == 1 }
        guard let s = (preferChunked ? specs.first(where: { $0.stages.count > 1 }) : nil) ?? specs.first else {
            throw HarnessError.noModel("no batch-1 UNet in the bundle")
        }
        spec = s
    }

    public var availableSlots: Set<PoseSlot> { Set(avatars.keys) }

    /// The cache for a slot, falling back the way a three-cache pose set does (smiling → idle → talking).
    public func bundle(_ slot: PoseSlot) -> AvatarBundle {
        avatars[FramePlan.resolve(slot, available: availableSlots)] ?? avatar
    }

    /// Clip cycle lengths per slot (the ping-pong cycle, as the latents and frames are indexed).
    public var cycleLengths: [PoseSlot: Int] {
        var out: [PoseSlot: Int] = [:]
        for (slot, b) in avatars { out[slot] = max(1, b.meta.cycle_length) }
        return out
    }

    /// Load the UNet and TAESD (the compile happens here on a first launch) and prebuild the per-frame latents.
    public func load() throws {
        let t0 = now()
        log("lipsync: loading \(spec.name) (\(spec.stages.count) stage(s)) on \(unitsName(units)) (\(memLine()))")
        unet = try harness.makeStagedUNet(spec, units: units)
        guard let tm = harness.meta.models(ofKind: "taesd").first(where: { $0.batch == 1 }) else {
            throw HarnessError.noModel("taesd_fp16_b1")
        }
        taesdMeta = tm
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        taesd = try MLModel(contentsOf: harness.resources.appendingPathComponent(tm.file), configuration: cfg)
        let ins = spec.inputs
        let sIn = ins.first { $0.name == "sample" } ?? ins[0]
        let aIn = ins.first { $0.name == "audio" } ?? ins[ins.count - 1]
        sampleName = sIn.name; audioName = aIn.name
        audioDType = try mlDataType(aIn.dtype)
        let sType = try mlDataType(sIn.dtype)
        var latentCount = 0
        for (slot, b) in avatars {
            sampleArrays[slot] = try (0 ..< b.meta.unique_frames).map { u in
                try makeArray(b.latents.frame(u), shape: [1] + b.latents.frameShape, dtype: sType)
            }
            latentCount += b.meta.unique_frames
        }
        loadSeconds = now() - t0
        log("lipsync: models loaded in \(String(format: "%.2f", loadSeconds)) s, \(latentCount) frame latents ready for \(avatars.keys.map { $0.rawValue }.sorted().joined(separator: "/")) (\(memLine()))")
    }

    /// The harness clip's 16 kHz waveform, when the bundle carries one.
    public func bundledWaveform() throws -> [Float] {
        guard let a = avatar.meta.audio else { throw HarnessError.noModel("this avatar bundle has no clip (phone-only)") }
        return try FlatTensor.load(TensorFile(file: a.waveform, shape: [1, a.samples], dtype: "float32"), in: avatar.dir).frame(0)
    }

    /// The harness clip's playable file, when the bundle carries one.
    public var playbackURL: URL? { avatar.meta.audio.map { avatar.dir.appendingPathComponent($0.playback) } }

    /// Run the bundled clip through log-mel + encoder + the window gather.
    public func prepareAudio() throws { try prepareAudio(waveform: try bundledWaveform()) }

    private var logmel: MLModel? = nil
    private var encoder: MLModel? = nil
    private var logmelMeta: ModelMeta? = nil
    private var encoderMeta: ModelMeta? = nil

    /// Run a 16 kHz mono waveform (any length) through log-mel + encoder + the window gather. The Whisper models are
    /// loaded on first use and kept (log-mel on the CPU, encoder on the Neural Engine).
    public func prepareAudio(waveform: [Float]) throws {
        let t0 = now()
        if logmel == nil || encoder == nil {
            guard let lm = harness.meta.models(ofKind: "whisper_logmel").first,
                  let em = harness.meta.models(ofKind: "whisper_encoder").first else {
                throw HarnessError.noModel("whisper models")
            }
            let cpu = MLModelConfiguration(); cpu.computeUnits = .cpuOnly
            let ne = MLModelConfiguration(); ne.computeUnits = .cpuAndNeuralEngine
            logmel = try MLModel(contentsOf: harness.resources.appendingPathComponent(lm.file), configuration: cpu)
            encoder = try MLModel(contentsOf: harness.resources.appendingPathComponent(em.file), configuration: ne)
            logmelMeta = lm; encoderMeta = em
        }
        windows = try computeAudioWindows(waveform: waveform, fps: avatar.fps, logmel: logmel!, logmelMeta: logmelMeta!,
                                          encoder: encoder!, encoderMeta: encoderMeta!)
        audioSeconds = now() - t0
        log("lipsync: audio windows for \(windows!.frames) frames at \(avatar.fps) fps (\(String(format: "%.1f", Double(waveform.count) / 16000)) s of audio) in \(String(format: "%.2f", audioSeconds)) s")
    }

    /// Max / mean abs difference of the Swift windows against the repo's windows exported with the avatar.
    public func checkWindows() throws -> (maxAbs: Double, meanAbs: Double, frames: Int) {
        guard let w = windows else { throw HarnessError.noModel("windows not prepared") }
        guard let a = avatar.meta.audio else { throw HarnessError.noModel("no reference windows in a phone-only bundle") }
        let ref = try FlatTensor.load(a.windows_ref, in: avatar.dir)
        let n = min(w.frames, ref.frameCount)
        var mx = 0.0, sum = 0.0, cnt = 0
        for i in 0 ..< n {
            let r = ref.frame(i)
            let s = w.window(i)
            for (k, v) in s.enumerated() {
                let d = abs(Double(v) - Double(r[k]))
                if d > mx { mx = d }
                sum += d; cnt += 1
            }
        }
        return (mx, sum / Double(max(1, cnt)), n)
    }

    public var frameCount: Int { windows?.frames ?? 0 }

    /// Decoder output -> 256x256 RGB 8-bit (vae.py convention without the BGR flip).
    private func faceRGB(_ faces: MLMultiArray) -> [UInt8] {
        let f = floats(faces)  // [1, 3, 256, 256] in [0, 1]
        var rgb = [UInt8](repeating: 0, count: 256 * 256 * 3)
        let plane = 256 * 256
        for p in 0 ..< plane {
            for c in 0 ..< 3 {
                let v = (f[c * plane + p] * 255).rounded(.toNearestOrEven)
                rgb[p * 3 + c] = UInt8(max(0, min(255, v)))
            }
        }
        return rgb
    }

    /// Stage A (Neural Engine): UNet chunks + TAESD for clip position `i` of the talking cache -> the 256x256 RGB face.
    /// Safe to run concurrently with `composeFrame`: it touches only the models, the windows and the prebuilt latents.
    public func inferFace(_ i: Int) throws -> (face: [UInt8], unetMs: Double, decodeMs: Double) {
        try inferFace(audioFrame: i, slot: .talking, cycle: i)
    }

    /// Stage A for audio window `i` on clip position `cycle` of `slot`'s cache (a turn whose pose plan switches caches).
    public func inferFace(audioFrame i: Int, slot: PoseSlot, cycle: Int) throws -> (face: [UInt8], unetMs: Double, decodeMs: Double) {
        guard let unet = unet, let taesd = taesd, let tm = taesdMeta, let w = windows else {
            throw HarnessError.noModel("engine not loaded")
        }
        let resolved = FramePlan.resolve(slot, available: availableSlots)
        let b = bundle(resolved)
        guard let samples = sampleArrays[resolved], !samples.isEmpty else { throw HarnessError.noModel("no latents for \(resolved.rawValue)") }
        let u = b.uniqueIndex(cycle)
        let t0 = now()
        let audio = try makeArray(Array(w.window(i)), shape: [1, 50, 384], dtype: audioDType)
        let pred = try output(unet.predict(MLDictionaryFeatureProvider(dictionary: [sampleName: samples[u], audioName: audio])),
                              spec.outputName)
        let t1 = now()
        let latIn = tm.inputs[0]
        let faces = try output(taesd.prediction(from: MLDictionaryFeatureProvider(dictionary: [latIn.name: try convert(pred, to: try mlDataType(latIn.dtype))])),
                               tm.outputs.first?.name ?? "faces")
        let rgb = faceRGB(faces)
        let t2 = now()
        return (rgb, (t1 - t0) * 1000, (t2 - t1) * 1000)
    }

    /// Stage B (CPU): the talking cache's frame for clip position `i` with the face pasted in (JPEG decode on a cache miss, resize, blend).
    public func composeFrame(_ i: Int, face: [UInt8]) throws -> (RGBAImage, composeMs: Double) {
        try composeFrame(slot: .talking, cycle: i, face: face)
    }

    /// Stage B on clip position `cycle` of `slot`'s cache.
    public func composeFrame(slot: PoseSlot, cycle: Int, face: [UInt8]) throws -> (RGBAImage, composeMs: Double) {
        let b = bundle(slot)
        let u = b.uniqueIndex(cycle)
        let t0 = now()
        let frame = try b.frame(u)
        let plan = try b.plan(u)
        let img = compose(frame: frame, face256: face, plan: plan)
        return (img, (now() - t0) * 1000)
    }

    /// A cache's frame as it is, no lipsync (the idle track between turns).
    public func plainFrame(slot: PoseSlot, cycle: Int) throws -> RGBAImage {
        let b = bundle(slot)
        return try b.frame(b.uniqueIndex(cycle))
    }

    /// One composed frame for clip position `i` (both stages, sequentially).
    public func renderFrame(_ i: Int) throws -> (RGBAImage, FrameTiming) {
        let a = try inferFace(i)
        let b = try composeFrame(i, face: a.face)
        return (b.0, FrameTiming(unetMs: a.unetMs, decodeMs: a.decodeMs, composeMs: b.composeMs))
    }

    /// Mean abs pixel difference against the Python-composed check frames (whole frame, and inside the face box).
    public func checkFrames() throws -> [(frame: Int, frameMeanAbs: Double, faceMeanAbs: Double, maxAbs: Int)] {
        guard let ck = avatar.meta.check else { return [] }
        var out: [(Int, Double, Double, Int)] = []
        for i in ck.frames where i < frameCount {
            let (img, _) = try renderFrame(i)
            let ref = try decodeRGBA(avatar.dir.appendingPathComponent(String(format: "check/%04d_composed.png", i)))
            let u = avatar.uniqueIndex(i)
            let box = avatar.meta.coords[u]
            var sum = 0.0, cnt = 0, fsum = 0.0, fcnt = 0, mx = 0
            for y in 0 ..< img.height {
                for x in 0 ..< img.width {
                    let p = (y * img.width + x) * 4
                    for c in 0 ..< 3 {
                        let d = abs(Int(img.pixels[p + c]) - Int(ref.pixels[p + c]))
                        sum += Double(d); cnt += 1
                        if d > mx { mx = d }
                        if x >= box[0] && x < box[2] && y >= box[1] && y < box[3] { fsum += Double(d); fcnt += 1 }
                    }
                }
            }
            out.append((i, sum / Double(cnt), fsum / Double(max(1, fcnt)), mx))
        }
        return out
    }
}

extension NSLock {
    /// `NSLock.withLock` needs iOS 16; this runs on the Lingua app's iOS 15.1 floor.
    @inline(__always) func sync<T>(_ body: () throws -> T) rethrows -> T {
        lock(); defer { unlock() }
        return try body()
    }
}

// MARK: - two-stage pipeline

/// The frame loop split over two threads: the Neural Engine stage (`inferFace`: UNet chunks + TAESD) and the CPU stage
/// (`composeFrame`: JPEG decode + resize + blend), joined by a short queue, so a frame costs max(ANE, CPU) instead of
/// their sum. The owner paces it: `targetIndex` is the frame the audio clock has reached (nil while the clock is not
/// running); frames already behind it are skipped before the UNet (what the server does when it drops) or dropped
/// before the blend. `downstreamCount` is how many finished frames the owner still holds; the pipeline keeps at most
/// `aheadFrames` finished or in flight.
public final class LipsyncPipeline {
    public struct Stats {
        public var generated = 0        // faces inferred on the Neural Engine
        public var composed = 0         // frames composed and emitted
        public var skipped = 0          // never inferred: the clock had passed them
        public var droppedLate = 0      // inferred, then too late to compose
        public var crossfaded = 0       // emitted frames blended with the previous pose's anchor
        public var inferMsSum = 0.0
        public var composeMsSum = 0.0
        public var leadFrames = 1       // how far ahead of the audio clock the last catch-up aimed
        public var inferMs: Double { generated > 0 ? inferMsSum / Double(generated) : 0 }
        public var composeMs: Double { composed > 0 ? composeMsSum / Double(composed) : 0 }
        public init() {}
    }

    public let engine: LipsyncEngine
    public var aheadFrames: Int
    public var queueDepth = 3
    /// Clip frame duration; with `targetIndex` it sets how far ahead of the clock a catch-up aims.
    public var frameMs = 1000.0 / 24
    public var targetIndex: () -> Int? = { nil }
    public var downstreamCount: () -> Int = { 0 }
    public var log: (String) -> Void = { _ in }
    /// Which cache and clip position each live frame uses, and the crossfades (nil = the talking cache at position i).
    public var plan: FramePlan? = nil
    /// The frame on screen when the turn starts (the frozen idle frame): the anchor of the first segment's crossfade.
    public var seamAnchor: RGBAImage? = nil
    /// The last composed frame of the turn, for the seam back to the idle track.
    public private(set) var lastEmitted: RGBAImage? = nil

    private let lock = NSLock()
    private var queue: [(Int, [UInt8])] = []
    private var stats = Stats()
    private var recentMs = 0.0           // moving average of one frame's time through both stages
    private var stopped = false
    private var inferDone = false
    private var threads: [Thread] = []

    public init(engine: LipsyncEngine, aheadFrames: Int) {
        self.engine = engine
        self.aheadFrames = aheadFrames
    }

    public var snapshot: Stats { lock.lock(); defer { lock.unlock() }; return stats }
    public var queued: Int { lock.lock(); defer { lock.unlock() }; return queue.count }

    /// Frames to aim ahead of the clock when catching up: the pipeline's own latency in frames, plus one. A frame
    /// inferred *at* the clock comes out 1–2 frames late on an A14 and is dropped; aimed ahead, it is shown.
    private func leadFrames() -> Int {
        let ms = lock.sync { recentMs }
        return ms > 0 ? Int((ms / frameMs).rounded(.up)) + 1 : 1
    }

    private func observe(frameMs ms: Double) {
        lock.sync { recentMs = recentMs > 0 ? recentMs * 0.8 + ms * 0.2 : ms }
    }

    /// Starts both stages. `emit` runs on the CPU stage's thread with every composed frame; `done` once, when the
    /// last frame has left the pipeline (or after `cancel`).
    public func start(from first: Int = 0, count: Int? = nil,
                      emit: @escaping (Int, RGBAImage) -> Void, done: @escaping () -> Void) {
        let total = min(first + (count ?? engine.frameCount), engine.frameCount)
        let infer = Thread { [self] in
            var i = first
            while i < total {
                if lock.sync({ stopped }) { break }
                if let t = targetIndex() {
                    let lead = leadFrames()
                    if t + lead > i {
                        lock.sync { stats.skipped += t + lead - i; stats.leadFrames = lead }
                        i = t + lead
                        if i >= total { break }
                    }
                }
                if downstreamCount() + queued >= aheadFrames || queued >= queueDepth { usleep(1000); continue }
                do {
                    let e = plan?.entry(i)
                    let r = try engine.inferFace(audioFrame: i, slot: e?.slot ?? .talking, cycle: e?.cycle ?? i)
                    lock.sync {
                        queue.append((i, r.face))
                        stats.generated += 1
                        stats.inferMsSum += r.unetMs + r.decodeMs
                    }
                    observe(frameMs: r.unetMs + r.decodeMs + lock.sync { stats.composeMs })
                } catch {
                    log("infer \(i): \(error)")
                }
                i += 1
            }
            lock.sync { inferDone = true }
        }
        let compose = Thread { [self] in
            var anchor: RGBAImage? = nil          // the previous pose's last composed frame, frozen at a switch
            var anchorSegment = Int.min
            while true {
                let (item, finished): ((Int, [UInt8])?, Bool) = lock.sync {
                    (queue.isEmpty ? nil : queue.removeFirst(), stopped || (inferDone && queue.isEmpty))
                }
                guard let (i, face) = item else {
                    if finished { break }
                    usleep(500)
                    continue
                }
                if let t = targetIndex(), i < t {
                    lock.sync { stats.droppedLate += 1 }
                    continue
                }
                do {
                    let e = plan?.entry(i)
                    var (img, ms) = try engine.composeFrame(slot: e?.slot ?? .talking, cycle: e?.cycle ?? i, face: face)
                    var blended = false
                    if let e = e, let (k, n) = e.crossfade {
                        // the scheduler's crossfade: the old pose's last frame fades under the incoming frames
                        if anchorSegment != e.segmentStart {
                            anchorSegment = e.segmentStart
                            anchor = lastEmitted ?? seamAnchor
                        }
                        if let an = anchor {
                            let t0 = now()
                            img = crossfade(anchor: an, incoming: img, alpha: crossfadeAlpha(k, of: n))
                            ms += (now() - t0) * 1000
                            blended = true
                        }
                    }
                    lastEmitted = img
                    emit(i, img)
                    lock.sync { stats.composed += 1; stats.composeMsSum += ms; if blended { stats.crossfaded += 1 } }
                } catch {
                    log("compose \(i): \(error)")
                }
            }
            done()
        }
        infer.name = "lipsync.infer"; infer.qualityOfService = .userInteractive
        compose.name = "lipsync.compose"; compose.qualityOfService = .userInteractive
        threads = [infer, compose]
        infer.start()
        compose.start()
    }

    public func cancel() { lock.sync { stopped = true } }
}

// MARK: - staged UNet shared with the harness

public struct StagedUNet {
    public let spec: UNetSpec
    public let models: [MLModel]

    /// Run the stages back to back: each takes what it needs, by name, from the inputs and the earlier outputs.
    public func predict(_ p: MLFeatureProvider) throws -> MLFeatureProvider {
        if models.count == 1 { return try models[0].prediction(from: p) }
        var feats: [String: MLFeatureValue] = [:]
        for n in p.featureNames { feats[n] = p.featureValue(for: n) }
        var last: MLFeatureProvider = p
        for (i, m) in models.enumerated() {
            var dict: [String: MLFeatureValue] = [:]
            for n in m.modelDescription.inputDescriptionsByName.keys {
                guard let v = feats[n] else { throw HarnessError.missingOutput("stage \(i + 1) input '\(n)'") }
                dict[n] = v
            }
            last = try m.prediction(from: MLDictionaryFeatureProvider(dictionary: dict))
            for n in last.featureNames { feats[n] = last.featureValue(for: n) }
        }
        return last
    }
}

extension Harness {
    public func makeStagedUNet(_ spec: UNetSpec, units: MLComputeUnits) throws -> StagedUNet {
        var models: [MLModel] = []
        for stage in spec.stages {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = units
            models.append(try MLModel(contentsOf: resources.appendingPathComponent(stage.file), configuration: cfg))
        }
        return StagedUNet(spec: spec, models: models)
    }
}
