import CryptoKit
import Foundation

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

    /// Fetch a phone avatar bundle (`pack_phone_bundle.py`) into `root/<avatarId>/`. A directory that already holds
    /// `avatar.json` is reused without a download.
    public static func fetchAvatarBundle(tarURL: URL, sha256: String?, avatarId: String, root: URL,
                                         log: (String) -> Void = { _ in }) async throws -> URL {
        let fm = FileManager.default
        let dir = root.appendingPathComponent(avatarId)
        if fm.fileExists(atPath: dir.appendingPathComponent("avatar.json").path) {
            log("avatar \(avatarId): already on this device")
            return dir
        }
        let tar = root.appendingPathComponent("\(avatarId).tar")
        try await download(tarURL, to: tar, sha256: sha256, log: log)
        let t0 = now()
        try? fm.removeItem(at: dir)
        try TarArchive.extract(tar, to: dir)
        try? fm.removeItem(at: tar)
        guard fm.fileExists(atPath: dir.appendingPathComponent("avatar.json").path) else {
            throw BundleError.format("bundle \(avatarId) has no avatar.json")
        }
        log(String(format: "avatar %@: unpacked in %.1f s", avatarId, now() - t0))
        return dir
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
        public let export_commit: String?
        public let meta: String
        public let files: [ModelPackFile]
        public let total_bytes: Int?
    }

    /// Fetch a model pack into `dir` (which then loads like a HarnessBundle folder): the manifest, `meta.json`, and
    /// every tar whose unpacked model is not already present. `baseURL` is the folder the manifest lives in;
    /// `resolve` maps a manifest file name to the URL to fetch it from (presigned URLs), overriding `baseURL`.
    public static func fetchModelPack(manifestURL: URL, baseURL: URL? = nil, into dir: URL,
                                      resolve: ((String) -> URL?)? = nil,
                                      progress: ((String, Int, Int) -> Void)? = nil,
                                      log: (String) -> Void = { _ in }) async throws -> ModelPackManifest {
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = baseURL ?? manifestURL.deletingLastPathComponent()
        let source: (String) -> URL = { name in resolve?(name) ?? base.appendingPathComponent(name) }
        let manifestFile = dir.appendingPathComponent("manifest.json")
        try await download(manifestURL, to: manifestFile, log: log)
        let manifest = try JSONDecoder().decode(ModelPackManifest.self, from: Data(contentsOf: manifestFile))
        try await download(source(manifest.meta), to: dir.appendingPathComponent("meta.json"), log: log)
        for (i, f) in manifest.files.enumerated() {
            progress?(f.name, i, manifest.files.count)
            let target = dir.appendingPathComponent(f.name)
            if fm.fileExists(atPath: target.path) {
                log("model \(f.name): already on this device")
                continue
            }
            let local = dir.appendingPathComponent(f.file)
            try await download(source(f.file), to: local, sha256: f.sha256, log: log)
            if f.file.hasSuffix(".tar") {
                try TarArchive.extract(local, to: dir)
                try? fm.removeItem(at: local)
            }
        }
        progress?("done", manifest.files.count, manifest.files.count)
        return manifest
    }
}
