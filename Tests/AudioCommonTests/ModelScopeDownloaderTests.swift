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
}
