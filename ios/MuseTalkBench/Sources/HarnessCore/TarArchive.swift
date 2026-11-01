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
    public static func extract(_ tarURL: URL, to dir: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        guard let fh = FileHandle(forReadingAtPath: tarURL.path) else {
            throw BundleError.io("cannot open \(tarURL.lastPathComponent)")
        }
        defer { try? fh.close() }
        var longName: String? = nil
        while true {
            guard let header = try fh.read(upToCount: 512), header.count == 512 else { break }
            if header.allSatisfy({ $0 == 0 }) { break }  // end-of-archive block
            let name = longName ?? entryName(header)
            longName = nil
            let size = octal(header, 124, 12)
            let padded = (size + 511) / 512 * 512
            let type = header[header.startIndex + 156]
            switch type {
            case UInt8(ascii: "5"):
                try fm.createDirectory(at: safeURL(dir, name), withIntermediateDirectories: true)
                if padded > 0 { _ = try fh.read(upToCount: padded) }
            case UInt8(ascii: "L"):  // GNU long name: the next entry's name
                let data = try fh.read(upToCount: padded) ?? Data()
                longName = String(decoding: data.prefix(size).prefix { $0 != 0 }, as: UTF8.self)
            case 0, UInt8(ascii: "0"), UInt8(ascii: "7"):
                let out = try safeURL(dir, name)
                try fm.createDirectory(at: out.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: out)
                guard fm.createFile(atPath: out.path, contents: nil), let w = FileHandle(forWritingAtPath: out.path) else {
                    throw BundleError.io("cannot write \(name)")
                }
                var left = size
                while left > 0 {
                    let n = min(left, 1 << 20)
                    guard let chunk = try fh.read(upToCount: n), chunk.count == n else {
                        try? w.close()
                        throw BundleError.format("truncated entry \(name)")
                    }
                    try w.write(contentsOf: chunk)
                    left -= n
                }
                try w.close()
                if padded > size { _ = try fh.read(upToCount: padded - size) }
            default:  // pax headers ('x', 'g'), links, devices: skip the data
                if padded > 0 { _ = try fh.read(upToCount: padded) }
            }
        }
    }

    private static func safeURL(_ dir: URL, _ name: String) throws -> URL {
        let parts = name.split(separator: "/").map(String.init)
        guard !parts.contains(".."), !name.hasPrefix("/") else { throw BundleError.format("unsafe path \(name)") }
        return parts.reduce(dir) { $0.appendingPathComponent($1) }
    }

    private static func entryName(_ h: Data) -> String {
        let b = h.startIndex
        let name = String(decoding: h[b ..< b + 100].prefix { $0 != 0 }, as: UTF8.self)
        let magic = String(decoding: h[b + 257 ..< b + 262], as: UTF8.self)
        let prefix = magic == "ustar" ? String(decoding: h[b + 345 ..< b + 500].prefix { $0 != 0 }, as: UTF8.self) : ""
        return prefix.isEmpty ? name : prefix + "/" + name
    }

    private static func octal(_ h: Data, _ offset: Int, _ length: Int) -> Int {
        var value = 0, started = false
        for byte in h[h.startIndex + offset ..< h.startIndex + offset + length] {
            if byte >= 48 && byte <= 55 { value = value * 8 + Int(byte - 48); started = true }
            else if started { break }
        }
        return value
    }
}
