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
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        if offlineMode && requestedFilesExist(in: directory, additionalFiles: additionalFiles) {
            progressHandler?(1.0)
            return
        }

        let filesToDownload = try await listModelFiles(modelId: modelId)

        let patterns = requestedPatterns(additionalFiles: additionalFiles)
        let filteredFiles = filesToDownload.filter { filePath in
            patterns.contains { matches(filePath: filePath, pattern: $0) }
        }
        let totalFiles = max(filteredFiles.count, 1)
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

    static func matches(filePath: String, pattern: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: globPatternToRegex(pattern)) else {
            return filePath == pattern
        }
        let range = NSRange(filePath.startIndex..<filePath.endIndex, in: filePath)
        return regex.firstMatch(in: filePath, range: range) != nil
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

    private static func requestedPatterns(additionalFiles: [String]) -> [String] {
        var patterns: [String] = ["config.json"]
        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            patterns.append("*.safetensors")
            patterns.append("model.safetensors.index.json")
        }
        for file in additionalFiles where !patterns.contains(file) {
            patterns.append(file)
        }
        return patterns
    }

    private static func requestedFilesExist(in directory: URL, additionalFiles: [String]) -> Bool {
        let localFiles = listRelativeFiles(in: directory)
        let patterns = requestedPatterns(additionalFiles: additionalFiles)
        return patterns.allSatisfy { pattern in
            localFiles.contains { matches(filePath: $0, pattern: pattern) }
        }
    }

    private static func listRelativeFiles(in directory: URL) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files: [String] = []
        while let fileURL = enumerator.nextObject() as? URL {
            guard let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey]),
                  values.isRegularFile == true else {
                continue
            }
            files.append(fileURL.path.replacingOccurrences(of: directory.path + "/", with: ""))
        }
        return files
    }

    private static func globPatternToRegex(_ pattern: String) -> String {
        var regex = "^"
        var index = pattern.startIndex

        while index < pattern.endIndex {
            let char = pattern[index]
            let nextIndex = pattern.index(after: index)

            if char == "*" {
                if nextIndex < pattern.endIndex && pattern[nextIndex] == "*" {
                    regex += ".*"
                    index = pattern.index(after: nextIndex)
                } else {
                    regex += "[^/]*"
                    index = nextIndex
                }
                continue
            }

            if ".+?^${}()|[]\\".contains(char) {
                regex += "\\"
            }
            regex.append(char)
            index = nextIndex
        }

        regex += "$"
        return regex
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
