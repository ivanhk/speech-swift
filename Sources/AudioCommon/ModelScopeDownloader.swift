import Foundation
import os

public enum ModelScopeDownloader {

    private static let baseURL = "https://modelscope.cn/api/v1/models"
    private static let defaultRevision = "master"

    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = resolveBaseCacheDir(cacheDirName: cacheDirName)
        let fm = FileManager.default

        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if weightsExist(in: oldDir) {
            return oldDir
        }

        let newDir = base.appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(modelId.replacingOccurrences(of: "/", with: "_"), isDirectory: true)
        try fm.createDirectory(at: newDir, withIntermediateDirectories: true)
        return newDir
    }

    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            AudioLog.download.debug("Could not list directory \(directory.path): \(error)")
            contents = []
        }
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let filesToDownload = try await listModelFiles(modelId: modelId)

        var filesNeeded: Set<String> = ["config.json"]
        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            for file in filesToDownload where file.hasSuffix(".safetensors") {
                filesNeeded.insert(file)
            }
            if filesToDownload.contains("model.safetensors.index.json") {
                filesNeeded.insert("model.safetensors.index.json")
            }
        }
        for file in additionalFiles {
            filesNeeded.insert(file)
        }

        let filteredFiles = filesToDownload.filter { filesNeeded.contains($0) }
        let totalFiles = filteredFiles.count
        var completedFiles = 0

        for file in filteredFiles {
            let localURL = try validatedLocalPath(directory: directory, fileName: file)
            if !FileManager.default.fileExists(atPath: localURL.path) {
                try await downloadFile(
                    modelId: modelId,
                    filePath: file,
                    to: localURL
                )
            }
            completedFiles += 1
            progressHandler?(Double(completedFiles) / Double(totalFiles))
        }
    }

    public static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    public static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    private static func resolveBaseCacheDir(cacheDirName: String) -> URL {
        let fm = FileManager.default
        let root: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }
        return root.appendingPathComponent(cacheDirName, isDirectory: true)
    }

    private static func listModelFiles(modelId: String) async throws -> [String] {
        let listURL = try makeListURL(modelId: modelId)

        var request = URLRequest(url: listURL)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("Invalid response for model listing: \(modelId)")
        }

        guard httpResponse.statusCode == 200 else {
            throw DownloadError.failedToDownload("Failed to list model files: HTTP \(httpResponse.statusCode)")
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let dataDict = json?["Data"] as? [String: Any],
              let files = dataDict["Files"] as? [[String: Any]] else {
            throw DownloadError.failedToDownload("Invalid response format for model listing")
        }

        return files.compactMap { $0["Path"] as? String }
    }

    private static func downloadFile(
        modelId: String,
        filePath: String,
        to localURL: URL
    ) async throws {
        let downloadURL = try makeDownloadURL(modelId: modelId, filePath: filePath)

        var request = URLRequest(url: downloadURL)
        request.httpMethod = "GET"

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("Invalid response for file download: \(filePath)")
        }

        guard httpResponse.statusCode == 200 else {
            throw DownloadError.failedToDownload("Failed to download \(filePath): HTTP \(httpResponse.statusCode)")
        }

        let parentDir = localURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        FileManager.default.createFile(atPath: localURL.path, contents: nil)
        let fileHandle = try FileHandle(forWritingTo: localURL)

        defer {
            try? fileHandle.close()
        }

        for try await byte in asyncBytes {
            fileHandle.write(Data([byte]))
        }

        AudioLog.download.info("Downloaded \(filePath) to \(localURL.path)")
    }

    static func makeListURL(modelId: String) throws -> URL {
        let encodedModelId = modelId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? modelId
        let listURLString = "\(baseURL)/\(encodedModelId)/repo/files?Revision=\(defaultRevision)"
        guard let listURL = URL(string: listURLString) else {
            throw DownloadError.failedToDownload("Invalid URL for model listing: \(modelId)")
        }
        return listURL
    }

    static func makeDownloadURL(modelId: String, filePath: String) throws -> URL {
        let encodedModelId = modelId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? modelId
        let encodedFilePath = filePath.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? filePath
        let downloadURLString = "\(baseURL)/\(encodedModelId)/repo?Revision=\(defaultRevision)&FilePath=\(encodedFilePath)"
        guard let downloadURL = URL(string: downloadURLString) else {
            throw DownloadError.failedToDownload("Invalid download URL: \(modelId)/\(filePath)")
        }
        return downloadURL
    }
}
