import CryptoKit
import Foundation
import ImageIO

/// Downloads phone bundles (avatars, model packs) with a SHA-256 check and unpacks them.
public enum RemoteBundle {
    /// Download `url` to `dest` (replaced), verifying the SHA-256 when one is given. Returns the byte count.
    @discardableResult
    public static func download(_ url: URL, to dest: URL, sha256: String? = nil, log: (String) -> Void = { _ in }) async throws -> Int {
        let t0 = now()
        let (tmp, response) = try await URLSession.shared.download(from: url)
        if let http = response as? HTTPURLResponse, !(200 ..< 300).contains(http.statusCode) {
            try? FileManager.default.removeItem(at: tmp)
            throw BundleError.io("HTTP \(http.statusCode) for \(url.lastPathComponent)")
        }
        let fm = FileManager.default
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? fm.removeItem(at: dest)
        try fm.moveItem(at: tmp, to: dest)
        let bytes = (try? fm.attributesOfItem(atPath: dest.path)[.size] as? Int) ?? 0
        if let want = sha256?.lowercased(), !want.isEmpty {
            let got = try sha256Hex(of: dest)
            guard got == want else {
                try? fm.removeItem(at: dest)
                throw BundleError.checksum("sha256 mismatch for \(url.lastPathComponent): got \(got.prefix(12))…, want \(want.prefix(12))…")
            }
        }
        log(String(format: "downloaded %@: %.1f MB in %.1f s", url.lastPathComponent, Double(bytes) / 1e6, now() - t0))
        return bytes
    }

    public static func sha256Hex(of file: URL) throws -> String {
        guard let fh = FileHandle(forReadingAtPath: file.path) else { throw BundleError.io("cannot read \(file.lastPathComponent)") }
        defer { try? fh.close() }
        var hasher = SHA256()
        while let chunk = try fh.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func validAvatarId(_ id: String) -> Bool {
        !id.isEmpty && id.count <= 200 && id != "." && id != ".." && id.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) || "_.-".unicodeScalars.contains($0)
        }
    }

    /// A published digest selects an immutable directory. Switching the pointer cannot change an active session's files.
    public static func installedAvatarDirectory(avatarId: String, root: URL, sha256: String? = nil) -> URL? {
        guard validAvatarId(avatarId) else { return nil }
        let base = root.appendingPathComponent(avatarId)
        guard let data = try? Data(contentsOf: base.appendingPathComponent("current.json")),
              let pointer = try? JSONSerialization.jsonObject(with: data) as? [String: String],
              let digest = pointer["sha256"], digest.count == 64,
              digest.allSatisfy({ $0.isHexDigit }), sha256 == nil || sha256?.lowercased() == digest else { return nil }
        let dir = base.appendingPathComponent(digest)
        return verifyInstallation(dir, digest: digest) ? dir : nil
    }

    private static func verifyInstallation(_ dir: URL, digest: String) -> Bool {
        guard let receiptData = try? Data(contentsOf: dir.appendingPathComponent("installation.json")),
              let receipt = try? JSONSerialization.jsonObject(with: receiptData) as? [String: Any],
              receipt["sha256"] as? String == digest,
              let manifestDigest = receipt["manifest_sha256"] as? String,
              (try? sha256Hex(of: dir.appendingPathComponent("avatar.json"))) == manifestDigest,
              let sizes = receipt["files"] as? [String: Int], !sizes.isEmpty else { return false }
        for (name, size) in sizes {
            guard safeRelativePath(name), let attributes = try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(name).path),
                  attributes[.type] as? FileAttributeType == .typeRegular,
                  (attributes[.size] as? NSNumber)?.intValue == size else { return false }
        }
        return true
    }

    private static func safeRelativePath(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/") && !path.contains("\\") && !path.split(separator: "/").contains(where: { $0 == ".." || $0 == "." })
    }

    /// Validate every referenced file before an installation can become visible.
    public static func validateAvatarBundle(at dir: URL, avatarId: String) throws -> [String: Int] {
        let metaURL = dir.appendingPathComponent("avatar.json")
        let metaData = try Data(contentsOf: metaURL)
        guard let meta = try JSONSerialization.jsonObject(with: metaData) as? [String: Any],
              meta["avatar_id"] as? String == avatarId,
              let n = meta["unique_frames"] as? Int, (1...2000).contains(n),
              let fps = meta["fps"] as? Int, (1...60).contains(fps),
              let w = meta["frame_width"] as? Int, (1...2048).contains(w),
              let h = meta["frame_height"] as? Int, (1...2048).contains(h),
              let mw = meta["mask_width"] as? Int, (1...4096).contains(mw),
              let mh = meta["mask_height"] as? Int, (1...4096).contains(mh),
              let frames = meta["frames"] as? [String], frames.count == n,
              let masks = meta["masks"] as? [String], masks.count == n,
              let coords = meta["coords"] as? [[Int]], coords.count == n,
              let maskCoords = meta["mask_coords"] as? [[Int]], maskCoords.count == n,
              let cycle = meta["cycle_length"] as? Int,
              let pingPong = meta["ping_pong"] as? Bool, cycle == (pingPong ? 2 * n : n),
              let latent = meta["latents"] as? [String: Any], latent["dtype"] as? String == "float16",
              latent["shape"] as? [Int] == [n, 8, 32, 32],
              let latentFile = latent["file"] as? String, latentFile == "latents_f16.bin" else {
            throw BundleError.format("invalid avatar geometry or tensor layout")
        }
        var sizes = ["avatar.json": metaData.count]
        func record(_ name: String) throws -> URL {
            guard safeRelativePath(name), sizes[name] == nil else { throw BundleError.format("duplicate or unsafe avatar path") }
            let url = dir.appendingPathComponent(name)
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            guard attrs[.type] as? FileAttributeType == .typeRegular, let bytes = attrs[.size] as? NSNumber, bytes.intValue > 0 else {
                throw BundleError.format("invalid avatar file \(name)")
            }
            sizes[name] = bytes.intValue
            return url
        }
        for i in 0..<n {
            guard frames[i].hasPrefix("frames/"), masks[i].hasPrefix("masks/"),
                  coords[i].count == 4, maskCoords[i].count == 4 else { throw BundleError.format("invalid frame coordinate list") }
            let c = coords[i], m = maskCoords[i]
            guard c[0] >= 0, c[1] >= 0, c[2] > c[0], c[3] > c[1], c[2] <= w, c[3] <= h,
                  m[2] > m[0], m[3] > m[1], m[2] - m[0] <= 4096, m[3] - m[1] <= 4096, m[2] > 0, m[3] > 0, m[0] < w, m[1] < h else {
                throw BundleError.format("invalid avatar crop bounds")
            }
            if i == 0 && (m[2] - m[0] != mw || m[3] - m[1] != mh) { throw BundleError.format("first mask dimensions differ from metadata") }
            for (name, width, height) in [(frames[i], w, h), (masks[i], m[2] - m[0], m[3] - m[1])] {
                let url = try record(name)
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                      props[kCGImagePropertyPixelWidth] as? Int == width,
                      props[kCGImagePropertyPixelHeight] as? Int == height else { throw BundleError.format("avatar image dimensions differ") }
            }
        }
        let latents = try Data(contentsOf: record(latentFile), options: .mappedIfSafe)
        guard latents.count == n * 8 * 32 * 32 * 2 else { throw BundleError.format("avatar latent byte count differs") }
        let finite = latents.withUnsafeBytes { bytes -> Bool in
            let p = bytes.bindMemory(to: UInt8.self)
            for i in stride(from: 0, to: p.count, by: 2) {
                let value = UInt16(p[i]) | (UInt16(p[i + 1]) << 8)
                if value & 0x7c00 == 0x7c00 { return false }
            }
            return true
        }
        guard finite else { throw BundleError.format("avatar latents contain non-finite values") }
        guard let enumerator = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey]) else { throw BundleError.io("cannot enumerate avatar") }
        for case let file as URL in enumerator {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                let name = String(file.resolvingSymlinksInPath().path.dropFirst(dir.resolvingSymlinksInPath().path.count + 1))
                guard sizes[name] != nil else { throw BundleError.format("unreferenced avatar file: \(name)") }
            }
        }
        return sizes
    }

    /// Verify, extract and validate in staging; only then atomically publish a hash-keyed installation.
    public static func fetchAvatarBundle(tarURL: URL, sha256: String?, avatarId: String, root: URL, expectedBytes: Int? = nil,
                                         log: (String) -> Void = { _ in }) async throws -> URL {
        guard validAvatarId(avatarId), let digest = sha256?.lowercased(), digest.count == 64,
              digest.allSatisfy({ $0.isHexDigit }), expectedBytes == nil || ((expectedBytes ?? 0) > 0 && (expectedBytes ?? 0) <= 1_073_741_824) else {
            throw BundleError.format("avatar requires a valid identity, SHA-256 and bounded byte count")
        }
        if let installed = installedAvatarDirectory(avatarId: avatarId, root: root, sha256: digest) { return installed }
        let fm = FileManager.default
        let base = root.appendingPathComponent(avatarId)
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var excluded = base; try? excluded.setResourceValues(values)
        let staging = base.appendingPathComponent(".staging-" + UUID().uuidString)
        let archive = base.appendingPathComponent(UUID().uuidString + ".tar")
        defer { try? fm.removeItem(at: archive); try? fm.removeItem(at: staging) }
        let got = try await download(tarURL, to: archive, sha256: digest, log: log)
        guard got <= 1_073_741_824, expectedBytes == nil || got == expectedBytes else { throw BundleError.format("avatar archive byte count differs") }
        try TarArchive.extract(archive, to: staging, strict: true, maxExpandedBytes: 1_073_741_824)
        let sizes = try validateAvatarBundle(at: staging, avatarId: avatarId)
        let receipt: [String: Any] = ["sha256": digest, "manifest_sha256": try sha256Hex(of: staging.appendingPathComponent("avatar.json")), "files": sizes]
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: staging.appendingPathComponent("installation.json"), options: .atomic)
        let target = base.appendingPathComponent(digest)
        try commitAvatar(staging: staging, target: target, base: base, digest: digest)
        guard let installed = installedAvatarDirectory(avatarId: avatarId, root: root, sha256: digest) else { throw BundleError.format("avatar installation incomplete") }
        return installed
    }

    private static let avatarCommitLock = NSLock()
    private static func commitAvatar(staging: URL, target: URL, base: URL, digest: String) throws {
        avatarCommitLock.lock(); defer { avatarCommitLock.unlock() }
        let fm = FileManager.default
        if !verifyInstallation(target, digest: digest) {
            if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
            try fm.moveItem(at: staging, to: target)
        }
        try JSONSerialization.data(withJSONObject: ["sha256": digest]).write(to: base.appendingPathComponent("current.json"), options: .atomic)
    }

    /// One entry of a model-pack manifest (`pack_model_pack.py`).
    public struct ModelPackFile: Decodable {
        public let name: String
        public let file: String
        public let bytes: Int
        public let sha256: String
        public let kind: String?
    }

    public struct ModelPackManifest: Decodable {
        public let version: String
        public let format: String?
        public let unet: String?
        public let render_fps: Int?
        public let quality_profile: String?
        public let export_commit: String?
        public let meta: String
        public let files: [ModelPackFile]
        public let total_bytes: Int?
    }

    /// Matching receipt and complete extracted files are required; a directory alone never means installed.
    public static func modelFileInstalled(_ file: ModelPackFile, in dir: URL) -> Bool {
        guard safeRelativePath(file.name), !file.name.contains("/"),
              let data = try? Data(contentsOf: dir.appendingPathComponent(file.name + ".installed.json")),
              let receipt = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              receipt["sha256"] as? String == file.sha256,
              let sizes = receipt["files"] as? [String: Int], !sizes.isEmpty else { return false }
        for (name, size) in sizes {
            guard safeRelativePath(name), let attrs = try? FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(name).path),
                  attrs[.type] as? FileAttributeType == .typeRegular,
                  (attrs[.size] as? NSNumber)?.intValue == size else { return false }
        }
        return true
    }

    /// Each verified file/archive is staged and atomically installed before its completion receipt is written.
    public static func fetchModelPack(manifestURL: URL, baseURL: URL? = nil, into dir: URL,
                                      manifestSHA256: String? = nil,
                                      resolve: ((String) -> URL?)? = nil,
                                      progress: ((String, Int, Int) -> Void)? = nil,
                                      log: (String) -> Void = { _ in }) async throws -> ModelPackManifest {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = baseURL ?? manifestURL.deletingLastPathComponent()
        let source: (String) -> URL = { name in resolve?(name) ?? base.appendingPathComponent(name) }
        let manifestFile = dir.appendingPathComponent("manifest.json")
        try await download(manifestURL, to: manifestFile, sha256: manifestSHA256, log: log)
        let manifest = try JSONDecoder().decode(ModelPackManifest.self, from: Data(contentsOf: manifestFile))
        guard manifest.meta == "meta.json", !manifest.files.isEmpty, manifest.files.count <= 100,
              Set(manifest.files.map { $0.name }).count == manifest.files.count else { throw BundleError.format("invalid model manifest") }
        let metaTemp = dir.appendingPathComponent(".meta-" + UUID().uuidString)
        defer { try? fm.removeItem(at: metaTemp) }
        try await download(source(manifest.meta), to: metaTemp, log: log)
        _ = try JSONSerialization.jsonObject(with: Data(contentsOf: metaTemp))
        try Data(contentsOf: metaTemp).write(to: dir.appendingPathComponent("meta.json"), options: .atomic)
        for (i, f) in manifest.files.enumerated() {
            guard safeRelativePath(f.name), !f.name.contains("/"), safeRelativePath(f.file), !f.file.contains("/"),
                  f.bytes > 0, f.bytes <= 2_147_483_648, f.sha256.count == 64, f.sha256.allSatisfy({ $0.isHexDigit }) else {
                throw BundleError.format("invalid model manifest file")
            }
            progress?(f.name, i, manifest.files.count)
            if modelFileInstalled(f, in: dir) { log("model \(f.name): verified installation present"); continue }
            let staging = dir.appendingPathComponent(".model-" + UUID().uuidString)
            let archive = dir.appendingPathComponent(".download-" + UUID().uuidString)
            defer { try? fm.removeItem(at: staging); try? fm.removeItem(at: archive) }
            try fm.createDirectory(at: staging, withIntermediateDirectories: true)
            let got = try await download(source(f.file), to: archive, sha256: f.sha256, log: log)
            guard got == f.bytes else { throw BundleError.format("model download byte count differs") }
            if f.file.hasSuffix(".tar") {
                try TarArchive.extract(archive, to: staging, strict: true, maxExpandedBytes: 2_147_483_648)
                let entries = try fm.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil)
                guard entries.count == 1, entries[0].lastPathComponent == f.name else { throw BundleError.format("model archive root differs") }
            } else { try fm.moveItem(at: archive, to: staging.appendingPathComponent(f.name)) }
            let staged = staging.appendingPathComponent(f.name)
            let sizes = try modelFileSizes(staged: staged, staging: staging, file: f, bytes: got)
            try commitModel(staged: staged, dir: dir, file: f, sizes: sizes)
        }
        progress?("done", manifest.files.count, manifest.files.count)
        return manifest
    }

    private static func modelFileSizes(staged: URL, staging: URL, file: ModelPackFile, bytes: Int) throws -> [String: Int] {
        let fm = FileManager.default
        var sizes: [String: Int] = [:]
        if (try staged.resourceValues(forKeys: [.isDirectoryKey])).isDirectory == true {
            guard let e = fm.enumerator(at: staged, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else { throw BundleError.io("cannot enumerate staged model") }
            for case let item as URL in e {
                let attrs = try item.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
                if attrs.isRegularFile == true {
                    let name = String(item.resolvingSymlinksInPath().path.dropFirst(staging.resolvingSymlinksInPath().path.count + 1))
                    sizes[name] = attrs.fileSize ?? 0
                }
            }
        } else { sizes[file.name] = bytes }
        guard !sizes.isEmpty else { throw BundleError.format("model archive is empty") }
        return sizes
    }

    private static func commitModel(staged: URL, dir: URL, file: ModelPackFile, sizes: [String: Int]) throws {
        avatarCommitLock.lock(); defer { avatarCommitLock.unlock() }
        if modelFileInstalled(file, in: dir) { return }
        let fm = FileManager.default, target = dir.appendingPathComponent(file.name)
        try? fm.removeItem(at: dir.appendingPathComponent(file.name + ".installed.json"))
        if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
        try fm.moveItem(at: staged, to: target)
        try JSONSerialization.data(withJSONObject: ["sha256": file.sha256, "files": sizes], options: [.sortedKeys]).write(to: dir.appendingPathComponent(file.name + ".installed.json"), options: .atomic)
    }
}
