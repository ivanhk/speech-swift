import Foundation
import XCTest
@testable import AudioServer

final class MultipartParsingTests: XCTestCase {
    func testParseMultipartWithFileFirstPreservesAudioAndFields() throws {
        let boundary = "Boundary-12345"
        let audioData = Data([0x52, 0x49, 0x46, 0x46, 0x00, 0x01, 0x02, 0x03])
        let body = makeMultipartBody(
            boundary: boundary,
            parts: [
                MultipartPart(
                    name: "file",
                    body: audioData,
                    filename: "audio.wav",
                    contentType: "audio/wav"
                ),
                MultipartPart(name: "model", body: Data("whisper-1".utf8)),
                MultipartPart(name: "language", body: Data("zh".utf8)),
                MultipartPart(name: "response_format", body: Data("verbose_json".utf8))
            ]
        )

        let params = try RequestParams.parseMultipart(
            body,
            contentType: "multipart/form-data; boundary = \"\(boundary)\"; charset=utf-8"
        )

        XCTAssertEqual(params.audioData, audioData)
        XCTAssertEqual(params.string("model"), "whisper-1")
        XCTAssertEqual(params.string("language"), "zh")
        XCTAssertEqual(params.string("response_format"), "verbose_json")
    }

    func testParseMultipartWithoutFileLeavesAudioDataNil() throws {
        let boundary = "Boundary-67890"
        let body = makeMultipartBody(
            boundary: boundary,
            parts: [
                MultipartPart(name: "model", body: Data("whisper-1".utf8)),
                MultipartPart(name: "language", body: Data("en".utf8))
            ]
        )

        let params = try RequestParams.parseMultipart(
            body,
            contentType: "multipart/form-data; boundary=\(boundary)"
        )

        XCTAssertNil(params.audioData)
        XCTAssertEqual(params.string("model"), "whisper-1")
        XCTAssertEqual(params.string("language"), "en")
    }
}

final class TranscriptionsEndpointValidationTests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19385

    override class func setUp() {
        super.setUp()
        serverTask = Task {
            let server = AudioServer(host: "127.0.0.1", port: port)
            try await server.run()
        }
        Thread.sleep(forTimeInterval: 1.5)
    }

    override class func tearDown() {
        serverTask?.cancel()
        Thread.sleep(forTimeInterval: 0.5)
        super.tearDown()
    }

    func testTranscriptionsEndpointReturnsBadRequestWhenFileIsMissing() async throws {
        let boundary = "Boundary-missing-file"
        let body = makeMultipartBody(
            boundary: boundary,
            parts: [
                MultipartPart(name: "model", body: Data("whisper-1".utf8)),
                MultipartPart(name: "language", body: Data("en".utf8))
            ]
        )

        let url = URL(string: "http://127.0.0.1:\(Self.port)/v1/audio/transcriptions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body

        let (data, response) = try await URLSession.shared.data(for: request)
        let httpResponse = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(httpResponse.statusCode, 400)

        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(json["error"], "Missing audio file")
    }
}

private struct MultipartPart {
    let name: String
    let body: Data
    var filename: String? = nil
    var contentType: String? = nil
}

private func makeMultipartBody(boundary: String, parts: [MultipartPart]) -> Data {
    var data = Data()

    for part in parts {
        data.append(Data("--\(boundary)\r\n".utf8))

        var disposition = "Content-Disposition: form-data; name=\"\(part.name)\""
        if let filename = part.filename {
            disposition += "; filename=\"\(filename)\""
        }
        data.append(Data("\(disposition)\r\n".utf8))

        if let contentType = part.contentType {
            data.append(Data("Content-Type: \(contentType)\r\n".utf8))
        }

        data.append(Data("\r\n".utf8))
        data.append(part.body)
        data.append(Data("\r\n".utf8))
    }

    data.append(Data("--\(boundary)--\r\n".utf8))
    return data
}
