import Foundation

/// `meta.json` as written by scripts/ios/prepare_harness_bundle.py.
public struct BundleMeta: Decodable {
    public let created: String?
    public let repo_commit: String?
    public let gate: GateSpec
    public let models: [ModelMeta]
    public let unet: UNetData
    public let whisper: WhisperData?
    public let mac_reference: JSONValue?

    public static func load(from resources: URL) throws -> BundleMeta {
        let data = try Data(contentsOf: resources.appendingPathComponent("meta.json"))
        return try JSONDecoder().decode(BundleMeta.self, from: data)
    }

    public func models(ofKind kind: String) -> [ModelMeta] { models.filter { $0.kind == kind } }
}

public struct GateSpec: Codable {
    public let mae: Double
    public let max_abs: Double
    public let source: String?
}

public struct IOFeature: Decodable {
    public let name: String
    public let shape: [Int]
    public let dtype: String
}

public struct ModelMeta: Decodable {
    public let name: String
    public let file: String
    public let kind: String
    public let batch: Int
    public let variant: String?
    public let layout: String?
    public let mlpackage_bytes: Int?
    public let chunk_of: String?
    public let chunk_index: Int?
    public let inputs: [IOFeature]
    public let outputs: [IOFeature]

    public func input(named name: String) -> IOFeature? { inputs.first { $0.name == name } }
}

public struct TensorFile: Decodable {
    public let file: String
    public let shape: [Int]
    public let dtype: String
    public let bytes: Int?

    public init(file: String, shape: [Int], dtype: String, bytes: Int? = nil) {
        self.file = file; self.shape = shape; self.dtype = dtype; self.bytes = bytes
    }
}

public struct CaptureFile: Decodable {
    public let name: String
    public let start: Int
    public let count: Int
    public let avatar: String?
}

public struct UNetData: Decodable {
    public let frames: Int
    public let capture_dir: String?
    public let latents: TensorFile
    public let audio: TensorFile
    public let ref: TensorFile
    public let files: [CaptureFile]
}

public struct WhisperData: Decodable {
    public let audio: String?
    public let speech_seconds: Double?
    public let waveform: TensorFile
    public let mel: TensorFile
    public let features_ref: TensorFile
}

/// Arbitrary JSON, so the Mac reference numbers pass through to the report untouched.
public enum JSONValue: Codable {
    case string(String), number(Double), bool(Bool), null
    case array([JSONValue]), object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): try c.encode(n)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    public subscript(key: String) -> JSONValue? {
        if case .object(let o) = self { return o[key] }
        return nil
    }

    public var double: Double? { if case .number(let n) = self { return n }; return nil }
}
