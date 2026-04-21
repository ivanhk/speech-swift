import XCTest
@testable import Qwen3ASR
@testable import KokoroTTS
@testable import AudioCommon

final class E2EForcedAlignerSentenceLevelTests: XCTestCase {

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

    func testChineseSentenceLevelAlignmentDefaultsToCharacterUnits() async throws {
        let text = "你好世界。这是一个测试。"
        let sentences = try await sentenceAlign(text: text, voice: "zf_xiaobei", language: "zh", granularity: .automatic)

        XCTAssertEqual(sentences.map(\.text), ["你好世界。", "这是一个测试。"])
        assertMonotonic(sentences, expectedCount: 2)
    }

    func testChineseSentenceLevelAlignmentSupportsWordLevel() async throws {
        let text = "你好世界。这是一个测试。"
        let sentences = try await sentenceAlign(text: text, voice: "zf_xiaobei", language: "zh", granularity: .word)

        XCTAssertEqual(sentences.map(\.text), ["你好世界。", "这是一个测试。"])
        assertMonotonic(sentences, expectedCount: 2)
    }

    func testEnglishSentenceLevelAlignmentOutputsSentences() async throws {
        let text = "Hello world. This is a test."
        let sentences = try await sentenceAlign(text: text, voice: "af_heart", language: "en", granularity: .automatic)

        XCTAssertEqual(sentences.map(\.text), ["Hello world.", "This is a test."])
        assertMonotonic(sentences, expectedCount: 2)
    }

    private func sentenceAlign(
        text: String,
        voice: String,
        language: String,
        granularity: ForcedAlignmentGranularity
    ) async throws -> [AlignedWord] {
        let tts = try await ttsModel()
        let audio = try tts.synthesize(text: text, voice: voice, language: language)
        let aligner = try await alignerModel()
        let alignedUnits = aligner.align(
            audio: audio,
            text: text,
            sampleRate: 24000,
            language: language,
            granularity: granularity
        )

        let sentences = TextPreprocessor.aggregateAlignedUnitsIntoSentences(
            alignedUnits,
            originalText: text,
            language: language,
            granularity: granularity
        )

        let duration = Float(audio.count) / 24000.0
        XCTAssertLessThanOrEqual(sentences.last?.endTime ?? 0, duration + 1.0)
        return sentences
    }

    private func assertMonotonic(_ aligned: [AlignedWord], expectedCount: Int) {
        XCTAssertEqual(aligned.count, expectedCount)
        XCTAssertFalse(aligned.isEmpty)

        for (index, sentence) in aligned.enumerated() {
            XCTAssertGreaterThanOrEqual(sentence.startTime, 0)
            XCTAssertGreaterThanOrEqual(sentence.endTime, sentence.startTime)

            if index > 0 {
                XCTAssertGreaterThanOrEqual(sentence.startTime, aligned[index - 1].startTime)
            }
        }
    }
}
