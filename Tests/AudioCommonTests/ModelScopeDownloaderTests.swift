import XCTest
@testable import AudioCommon

final class ModelScopeDownloaderTests: XCTestCase {

    func testListURLUsesRepoFilesEndpoint() throws {
        let url = try ModelScopeDownloader.makeListURL(modelId: "aufklarer/Qwen3-ASR-1.7B-MLX-4bit")

        XCTAssertEqual(url.host(), "modelscope.cn")
        XCTAssertEqual(url.path(), "/api/v1/models/aufklarer/Qwen3-ASR-1.7B-MLX-4bit/repo/files")
        XCTAssertEqual(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.first(where: { $0.name == "Revision" })?.value, "master")
    }

    func testDownloadURLKeepsRepoEndpointAndEncodesFilePath() throws {
        let url = try ModelScopeDownloader.makeDownloadURL(
            modelId: "aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit",
            filePath: "tokenizer.json"
        )

        XCTAssertEqual(url.path(), "/api/v1/models/aufklarer/Qwen3-TTS-12Hz-1.7B-Base-MLX-8bit/repo")
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        XCTAssertEqual(components.queryItems?.first(where: { $0.name == "Revision" })?.value, "master")
        XCTAssertEqual(components.queryItems?.first(where: { $0.name == "FilePath" })?.value, "tokenizer.json")
    }

    func testOfflineModeReturnsImmediatelyWhenWeightsExist() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        FileManager.default.createFile(
            atPath: tempDir.appendingPathComponent("model.safetensors").path,
            contents: Data()
        )

        var reportedProgress: Double?
        try await ModelScopeDownloader.downloadWeights(
            modelId: "org/offline-model",
            to: tempDir,
            offlineMode: true,
            progressHandler: { reportedProgress = $0 }
        )

        XCTAssertEqual(reportedProgress, 1.0)
    }

    func testGlobPatternsMatchNestedModelScopeFiles() {
        XCTAssertTrue(ModelScopeDownloader.matches(filePath: "voices/af_heart.json", pattern: "voices/*.json"))
        XCTAssertTrue(ModelScopeDownloader.matches(
            filePath: "kokoro_21_5s.mlmodelc/Data/com.apple.CoreML/model.mil",
            pattern: "kokoro_21_5s.mlmodelc/**"
        ))
        XCTAssertFalse(ModelScopeDownloader.matches(filePath: "voices/af_heart.bin", pattern: "voices/*.json"))
    }
}
