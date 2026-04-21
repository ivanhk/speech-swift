import XCTest
@testable import Qwen3ASR
@testable import KokoroTTS
@testable import AudioCommon

final class E2EForcedAlignerChineseTests: XCTestCase {

    private static var sharedTTS: KokoroTTSModel?
    private static var sharedAligner: Qwen3ForcedAligner?

    private func ttsModel() async throws -> KokoroTTSModel {
        if let model = Self.sharedTTS { return model }
        let model = try await KokoroTTSModel.fromPretrained()
        Self.sharedTTS = model
        return model
    }

    private func alignerModel() async throws -> Qwen3ForcedAligner {
        if let model = Self.sharedAligner { return model }
        let model = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit"
        )
        Self.sharedAligner = model
        return model
    }

    func testChineseAlignmentDefaultsToCharacterLevel() async throws {
        let text = "你好世界，这是一个测试。"
        let audio = try await synthesizedChineseAudio(for: text)
        let aligner = try await alignerModel()

        let aligned = aligner.align(
            audio: audio,
            text: text,
            sampleRate: 24000,
            language: "zh",
            granularity: .automatic
        )

        XCTAssertEqual(aligned.map(\.text), ["你", "好", "世", "界", "这", "是", "一", "个", "测", "试"])
        assertMonotonic(aligned, audioSampleCount: audio.count, sampleRate: 24000)
    }

    func testChineseAlignmentSupportsWordLevel() async throws {
        let text = "你好世界，这是一个测试。"
        let audio = try await synthesizedChineseAudio(for: text)
        let aligner = try await alignerModel()

        let charAligned = aligner.align(
            audio: audio,
            text: text,
            sampleRate: 24000,
            language: "zh",
            granularity: .char
        )
        let wordAligned = aligner.align(
            audio: audio,
            text: text,
            sampleRate: 24000,
            language: "zh",
            granularity: .word
        )

        XCTAssertEqual(wordAligned.map(\.text).joined(), "你好世界这是一个测试")
        XCTAssertLessThan(wordAligned.count, charAligned.count)
        assertMonotonic(wordAligned, audioSampleCount: audio.count, sampleRate: 24000)
    }

    private func synthesizedChineseAudio(for text: String) async throws -> [Float] {
        let model = try await ttsModel()
        return try model.synthesize(text: text, voice: "zf_xiaobei", language: "zh")
    }

    private func assertMonotonic(_ aligned: [AlignedWord], audioSampleCount: Int, sampleRate: Int) {
        XCTAssertFalse(aligned.isEmpty)
        let duration = Float(audioSampleCount) / Float(sampleRate)

        for (index, word) in aligned.enumerated() {
            XCTAssertGreaterThanOrEqual(word.startTime, 0)
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime)
            XCTAssertLessThanOrEqual(word.endTime, duration + 1.0)

            if index > 0 {
                XCTAssertGreaterThanOrEqual(word.startTime, aligned[index - 1].startTime)
            }
        }
    }
}
