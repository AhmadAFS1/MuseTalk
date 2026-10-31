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
    public let audio: AvatarAudioMeta
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
    public var totalFrames: Int { meta.audio.frames }

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

/// waveform (16 kHz, <= 30 s) -> per-frame UNet audio windows, through the Core ML log-mel and encoder.
public func computeAudioWindows(waveform: [Float], fps: Int, logmel: MLModel, logmelMeta: ModelMeta,
                                encoder: MLModel, encoderMeta: ModelMeta) throws -> AudioWindows {
    let segSamples = 16000 * 30
    var seg = [Float](repeating: 0, count: segSamples)
    let n = min(waveform.count, segSamples)
    seg.replaceSubrange(0 ..< n, with: waveform[0 ..< n])
    let lmIn = logmelMeta.inputs[0]
    let wave = try makeArray(seg, shape: lmIn.shape, dtype: try mlDataType(lmIn.dtype))
    let mel = try output(logmel.prediction(from: MLDictionaryFeatureProvider(dictionary: [lmIn.name: wave])),
                         logmelMeta.outputs.first?.name ?? "logmel")
    let encIn = encoderMeta.inputs[0]
    let melIn = try convert(mel, to: try mlDataType(encIn.dtype))
    let feat = try output(encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: [encIn.name: melIn])),
                          encoderMeta.outputs.first?.name ?? "features")
    let features = floats(feat)  // [1500, 5, 384]
    let layers = 5, dim = 384, steps = 10
    let actual = Int((Double(n) / 16000.0 * 50.0).rounded(.down))
    let frames = Int((Double(n) / 16000.0 * Double(fps)).rounded(.down))
    let padL = 4, padR = 12
    let total = padL + actual + padR
    var data = [Float](repeating: 0, count: frames * steps * layers * dim)
    data.withUnsafeMutableBufferPointer { out in
        features.withUnsafeBufferPointer { f in
            for i in 0 ..< frames {
                let a = (i * 50) / fps
                for s in 0 ..< steps {
                    let t = a + s  // index into the padded timeline
                    let src = t - padL
                    guard t < total, src >= 0, src < actual, src < 1500 else { continue }  // zero padding
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
    public let avatar: AvatarBundle
    public let spec: UNetSpec
    public let units: MLComputeUnits
    public var log: (String) -> Void = { print($0) }

    private var unet: StagedUNet? = nil
    private var taesd: MLModel? = nil
    private var taesdMeta: ModelMeta? = nil
    private var sampleArrays: [MLMultiArray] = []
    private var sampleName = "sample", audioName = "audio"
    private var audioDType: MLMultiArrayDataType = .float16
    public private(set) var windows: AudioWindows? = nil
    public private(set) var loadSeconds = 0.0
    public private(set) var audioSeconds = 0.0

    public init(resources: URL, avatarDir: URL, units: MLComputeUnits = .cpuAndNeuralEngine, preferChunked: Bool = true) throws {
        harness = try Harness(resources: resources)
        avatar = try AvatarBundle(dir: avatarDir)
        self.units = units
        let specs = harness.unetSpecs().filter { $0.batch == 1 }
        guard let s = (preferChunked ? specs.first(where: { $0.stages.count > 1 }) : nil) ?? specs.first else {
            throw HarnessError.noModel("no batch-1 UNet in the bundle")
        }
        spec = s
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
        sampleArrays = try (0 ..< avatar.meta.unique_frames).map { u in
            try makeArray(avatar.latents.frame(u), shape: [1] + avatar.latents.frameShape, dtype: sType)
        }
        loadSeconds = now() - t0
        log("lipsync: models loaded in \(String(format: "%.2f", loadSeconds)) s, \(sampleArrays.count) frame latents ready (\(memLine()))")
    }

    /// Run the clip's audio through log-mel + encoder + the window gather.
    public func prepareAudio() throws {
        let t0 = now()
        let a = avatar.meta.audio
        let wave = try FlatTensor.load(TensorFile(file: a.waveform, shape: [1, a.samples], dtype: "float32"),
                                       in: avatar.dir).frame(0)
        guard let lm = harness.meta.models(ofKind: "whisper_logmel").first,
              let em = harness.meta.models(ofKind: "whisper_encoder").first else {
            throw HarnessError.noModel("whisper models")
        }
        let cpu = MLModelConfiguration(); cpu.computeUnits = .cpuOnly
        let ne = MLModelConfiguration(); ne.computeUnits = .cpuAndNeuralEngine
        let logmel = try MLModel(contentsOf: harness.resources.appendingPathComponent(lm.file), configuration: cpu)
        let encoder = try MLModel(contentsOf: harness.resources.appendingPathComponent(em.file), configuration: ne)
        windows = try computeAudioWindows(waveform: wave, fps: avatar.fps, logmel: logmel, logmelMeta: lm, encoder: encoder, encoderMeta: em)
        audioSeconds = now() - t0
        log("lipsync: audio windows for \(windows!.frames) frames at \(avatar.fps) fps in \(String(format: "%.2f", audioSeconds)) s")
    }

    /// Max / mean abs difference of the Swift windows against the repo's windows exported with the avatar.
    public func checkWindows() throws -> (maxAbs: Double, meanAbs: Double, frames: Int) {
        guard let w = windows else { throw HarnessError.noModel("windows not prepared") }
        let ref = try FlatTensor.load(avatar.meta.audio.windows_ref, in: avatar.dir)
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

    /// One composed frame for clip position `i`.
    public func renderFrame(_ i: Int) throws -> (RGBAImage, FrameTiming) {
        guard let unet = unet, let taesd = taesd, let tm = taesdMeta, let w = windows else {
            throw HarnessError.noModel("engine not loaded")
        }
        let u = avatar.uniqueIndex(i)
        let t0 = now()
        let audio = try makeArray(Array(w.window(i)), shape: [1, 50, 384], dtype: audioDType)
        let pred = try output(unet.predict(MLDictionaryFeatureProvider(dictionary: [sampleName: sampleArrays[u], audioName: audio])),
                              spec.outputName)
        let t1 = now()
        let latIn = tm.inputs[0]
        let faces = try output(taesd.prediction(from: MLDictionaryFeatureProvider(dictionary: [latIn.name: try convert(pred, to: try mlDataType(latIn.dtype))])),
                               tm.outputs.first?.name ?? "faces")
        let rgb = faceRGB(faces)
        let t2 = now()
        let frame = try avatar.frame(u)
        let plan = try avatar.plan(u)
        let img = compose(frame: frame, face256: rgb, plan: plan)
        let t3 = now()
        return (img, FrameTiming(unetMs: (t1 - t0) * 1000, decodeMs: (t2 - t1) * 1000, composeMs: (t3 - t2) * 1000))
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
