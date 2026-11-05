import Foundation

/// Errors from downloading and unpacking phone bundles.
public enum BundleError: Error, CustomStringConvertible {
    case io(String)
    case format(String)
    case checksum(String)
    case busy(String)

    public var description: String {
        switch self {
        case .io(let s), .format(let s), .checksum(let s), .busy(let s): return s
        }
    }
}

/// Streaming reader for the plain ustar tars that `pack_phone_bundle.py` and `pack_model_pack.py` write (no
/// compression: frames are JPEG, masks PNG, weights int8/fp16 blobs). Reads 512-byte headers and copies file data in
/// 1 MB chunks, so a 486 MB model tar never sits in memory. Skips pax/GNU extension entries and links.
public enum TarArchive {
    public static func extract(_ tarURL: URL, to dir: URL, strict: Bool = false, maxExpandedBytes: Int = 4_294_967_296) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        guard let fh = FileHandle(forReadingAtPath: tarURL.path) else { throw BundleError.io("cannot open archive") }
        defer { try? fh.close() }
        var longName: String? = nil
        var total = 0
        var entries = Set<String>()
        func readExact(_ count: Int) throws -> Data {
            guard let data = try fh.read(upToCount: count), data.count == count else { throw BundleError.format("truncated archive") }
            return data
        }
        func skip(_ count: Int) throws {
            var left = count
            while left > 0 { let n = min(left, 1 << 20); _ = try readExact(n); left -= n }
        }
        while true {
            let header = try readExact(512)
            if header.allSatisfy({ $0 == 0 }) { break }
            var checksum = 0
            for i in 0..<512 { checksum += (148..<156).contains(i) ? 32 : Int(header[i]) }
            guard try octal(header, 148, 8) == checksum else { throw BundleError.format("archive checksum differs") }
            let name = longName ?? entryName(header)
            longName = nil
            let size = try octal(header, 124, 12)
            guard size <= maxExpandedBytes - total, entries.count < 100_000 else { throw BundleError.format("archive exceeds size or entry limit") }
            total += size
            let padded = (size + 511) / 512 * 512
            let type = header[156]
            if strict {
                guard type == 0 || type == UInt8(ascii: "0") || type == UInt8(ascii: "5"), entries.insert(name).inserted else {
                    throw BundleError.format("unsupported or duplicate avatar archive entry")
                }
            }
            switch type {
            case UInt8(ascii: "5"):
                try fm.createDirectory(at: safeURL(dir, name), withIntermediateDirectories: true)
                try skip(padded)
            case UInt8(ascii: "L"):
                guard size <= 4096 else { throw BundleError.format("archive long path exceeds limit") }
                let data = try readExact(padded)
                longName = String(decoding: data.prefix(size).prefix { $0 != 0 }, as: UTF8.self)
            case 0, UInt8(ascii: "0"), UInt8(ascii: "7"):
                let out = try safeURL(dir, name)
                try fm.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
                if strict && fm.fileExists(atPath: out.path) { throw BundleError.format("avatar file already exists") }
                try? fm.removeItem(at: out)
                guard fm.createFile(atPath: out.path, contents: nil), let w = FileHandle(forWritingAtPath: out.path) else { throw BundleError.io("cannot write archive entry") }
                do {
                    var left = size
                    while left > 0 {
                        let n = min(left, 1 << 20)
                        try w.write(contentsOf: readExact(n))
                        left -= n
                    }
                    try w.close()
                } catch { try? w.close(); throw error }
                try skip(padded - size)
            default:
                // Model packs can contain metadata; strict avatar archives reject it above.
                try skip(padded)
            }
        }
    }

    private static func safeURL(_ dir: URL, _ name: String) throws -> URL {
        let parts = name.split(separator: "/").map(String.init)
        guard !parts.isEmpty, !parts.contains(".."), !parts.contains("."), !name.hasPrefix("/"), !name.contains("\\") else { throw BundleError.format("unsafe path \(name)") }
        return parts.reduce(dir) { $0.appendingPathComponent($1) }
    }

    private static func entryName(_ h: Data) -> String {
        let b = h.startIndex
        let name = String(decoding: h[b ..< b + 100].prefix { $0 != 0 }, as: UTF8.self)
        let magic = String(decoding: h[b + 257 ..< b + 262], as: UTF8.self)
        let prefix = magic == "ustar" ? String(decoding: h[b + 345 ..< b + 500].prefix { $0 != 0 }, as: UTF8.self) : ""
        return prefix.isEmpty ? name : prefix + "/" + name
    }

    private static func octal(_ h: Data, _ offset: Int, _ length: Int) throws -> Int {
        var value = 0
        for byte in h[h.startIndex + offset ..< h.startIndex + offset + length] {
            if byte == 0 || byte == 32 { continue }
            guard byte >= 48 && byte <= 55, value <= (Int.max - 7) / 8 else { throw BundleError.format("invalid archive number") }
            value = value * 8 + Int(byte - 48)
        }
        return value
    }
}
