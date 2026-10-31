import CoreML
import Foundation

public enum HarnessError: Error, CustomStringConvertible {
    case badTensorSize(String)
    case unsupportedDType(String)
    case missingOutput(String)
    case noModel(String)

    public var description: String {
        switch self {
        case .badTensorSize(let s): return "tensor size mismatch: \(s)"
        case .unsupportedDType(let s): return "unsupported dtype: \(s)"
        case .missingOutput(let s): return "model returned no output named \(s)"
        case .noModel(let s): return "no model in the bundle: \(s)"
        }
    }
}

/// A flat binary tensor exported by prepare_harness_bundle.py: frames along dim 0.
public struct FlatTensor {
    public let shape: [Int]
    public let dtype: String
    public let data: Data

    public var frameCount: Int { shape[0] }
    public var frameShape: [Int] { Array(shape.dropFirst()) }
    public var frameElements: Int { frameShape.reduce(1, *) }
    public var elementSize: Int { dtype == "float16" ? 2 : 4 }

    public static func load(_ tf: TensorFile, in dir: URL) throws -> FlatTensor {
        let data = try Data(contentsOf: dir.appendingPathComponent(tf.file), options: .mappedIfSafe)
        let t = FlatTensor(shape: tf.shape, dtype: tf.dtype, data: data)
        guard data.count == tf.shape.reduce(1, *) * t.elementSize else {
            throw HarnessError.badTensorSize("\(tf.file): \(data.count) bytes for shape \(tf.shape) \(tf.dtype)")
        }
        return t
    }

    /// One frame as Float32.
    public func frame(_ i: Int) -> [Float] {
        let n = frameElements
        let start = i * n * elementSize
        return data.withUnsafeBytes { raw -> [Float] in
            let slice = UnsafeRawBufferPointer(rebasing: raw[start ..< start + n * elementSize])
            switch dtype {
            case "float32":
                return Array(slice.bindMemory(to: Float.self))
            case "float16":
                return float16ToFloat(slice, count: n)
            default:
                return []
            }
        }
    }

    public func frames(_ idx: [Int]) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(idx.count * frameElements)
        for i in idx { out.append(contentsOf: frame(i)) }
        return out
    }
}

// MARK: - half precision

func float16ToFloat(_ raw: UnsafeRawBufferPointer, count: Int) -> [Float] {
    #if arch(arm64)
    let p = raw.bindMemory(to: Float16.self)
    return (0 ..< count).map { Float(p[$0]) }
    #else
    // Software decode for x86_64 (Rosetta / Intel): sign, exponent, mantissa.
    let p = raw.bindMemory(to: UInt16.self)
    return (0 ..< count).map { i in
        let h = p[i]
        let sign = Float((h & 0x8000) != 0 ? -1 : 1)
        let exp = Int((h >> 10) & 0x1F)
        let mant = Float(h & 0x3FF)
        if exp == 0 { return sign * mant * powf(2, -24) }
        if exp == 31 { return mant == 0 ? sign * Float.infinity : Float.nan }
        return sign * (1 + mant / 1024) * powf(2, Float(exp - 15))
    }
    #endif
}

func writeFloat16(_ values: [Float], to raw: UnsafeMutableRawBufferPointer) {
    #if arch(arm64)
    let p = raw.bindMemory(to: Float16.self)
    for (i, v) in values.enumerated() { p[i] = Float16(v) }
    #else
    let p = raw.bindMemory(to: UInt16.self)
    for (i, v) in values.enumerated() {
        // round-to-nearest-even software encode, enough for test inputs
        let bits = v.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        var exp = Int((bits >> 23) & 0xFF) - 127 + 15
        var mant = bits & 0x7FFFFF
        if exp <= 0 { p[i] = sign; continue }
        if exp >= 31 { p[i] = sign | 0x7C00; continue }
        mant = mant >> 13
        if exp < 0 { exp = 0 }
        p[i] = sign | UInt16(exp << 10) | UInt16(mant)
    }
    #endif
}

// MARK: - MLMultiArray helpers

public func mlDataType(_ dtype: String) throws -> MLMultiArrayDataType {
    switch dtype {
    case "float16": return .float16
    case "float32": return .float32
    case "float64": return .double
    case "int32": return .int32
    default: throw HarnessError.unsupportedDType(dtype)
    }
}

/// Build an input array of `shape` and `dtype` from Float32 values.
public func makeArray(_ values: [Float], shape: [Int], dtype: MLMultiArrayDataType) throws -> MLMultiArray {
    let n = shape.reduce(1, *)
    guard values.count == n else {
        throw HarnessError.badTensorSize("\(values.count) values for shape \(shape)")
    }
    let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dtype)
    arr.withUnsafeMutableBytes { raw, _ in
        switch dtype {
        case .float16:
            writeFloat16(values, to: raw)
        case .float32:
            let p = raw.bindMemory(to: Float.self)
            for (i, v) in values.enumerated() { p[i] = v }
        case .double:
            let p = raw.bindMemory(to: Double.self)
            for (i, v) in values.enumerated() { p[i] = Double(v) }
        case .int32:
            let p = raw.bindMemory(to: Int32.self)
            for (i, v) in values.enumerated() { p[i] = Int32(v) }
        default:
            break
        }
    }
    return arr
}

private func isContiguous(_ arr: MLMultiArray) -> Bool {
    var expected = 1
    for d in stride(from: arr.shape.count - 1, through: 0, by: -1) {
        if arr.strides[d].intValue != expected { return false }
        expected *= arr.shape[d].intValue
    }
    return true
}

/// Read any MLMultiArray as Float32 (fast path for contiguous fp16/fp32/fp64).
public func floats(_ arr: MLMultiArray) -> [Float] {
    let n = arr.count
    if isContiguous(arr) {
        let out: [Float]? = arr.withUnsafeBytes { raw -> [Float]? in
            switch arr.dataType {
            case .float16: return float16ToFloat(raw, count: n)
            case .float32: return Array(raw.bindMemory(to: Float.self).prefix(n))
            case .double: return raw.bindMemory(to: Double.self).prefix(n).map { Float($0) }
            default: return nil
            }
        }
        if let out = out { return out }
    }
    return (0 ..< n).map { arr[$0].floatValue }
}

/// Re-type an array (e.g. fp32 UNet output -> fp16 TAESD input). Returns the input if it already matches.
public func convert(_ arr: MLMultiArray, to dtype: MLMultiArrayDataType) throws -> MLMultiArray {
    if arr.dataType == dtype { return arr }
    return try makeArray(floats(arr), shape: arr.shape.map { $0.intValue }, dtype: dtype)
}

public func output(_ provider: MLFeatureProvider, _ name: String) throws -> MLMultiArray {
    guard let arr = provider.featureValue(for: name)?.multiArrayValue else {
        throw HarnessError.missingOutput(name)
    }
    return arr
}
