import Foundation
import AudioCommon

/// Requested token grouping granularity for forced alignment text preprocessing.
public enum ForcedAlignmentGranularity: String, Sendable {
    case automatic
    case char
    case word
}

/// Result of preprocessing text for forced alignment
public struct SlottedText: Sendable {
    /// Token IDs with timestamp tokens inserted around each word
    public let tokenIds: [Int]
    /// Indices within tokenIds that are timestamp tokens
    public let timestampPositions: [Int]
    /// The original words (one per timestamp pair)
    public let words: [String]
}

/// Language-specific text preprocessing for forced alignment
public enum TextPreprocessor {

    /// Split text into words and insert timestamp slots for alignment.
    ///
    /// For each word, inserts `<timestamp><timestamp>` pairs so the model
    /// can predict start/end timestamps at those positions.
    ///
    /// - Parameters:
    ///   - text: Input text to align
    ///   - tokenizer: Tokenizer for encoding word tokens
    ///   - language: Language hint for word splitting strategy
    /// - Returns: SlottedText with token IDs, timestamp positions, and words
    public static func prepareForAlignment(
        text: String,
        tokenizer: Qwen3Tokenizer,
        language: String? = nil,
        granularity: ForcedAlignmentGranularity = .automatic
    ) -> SlottedText {
        let words = splitIntoWords(text, language: language, granularity: granularity)
        let tsId = Qwen3ASRTokens.timestampTokenId

        var tokenIds: [Int] = []
        var timestampPositions: [Int] = []
        var validWords: [String] = []

        for word in words {
            let wordTokens = tokenizer.encode(word)
            guard !wordTokens.isEmpty else { continue }

            // Insert <timestamp> before word (start marker)
            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            // Word tokens
            tokenIds.append(contentsOf: wordTokens)

            // Insert <timestamp> after word (end marker)
            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            validWords.append(word)
        }

        return SlottedText(
            tokenIds: tokenIds,
            timestampPositions: timestampPositions,
            words: validWords
        )
    }

    /// Split text into words using language-appropriate strategy
    static func splitIntoWords(
        _ text: String,
        language: String?,
        granularity: ForcedAlignmentGranularity = .automatic
    ) -> [String] {
        if shouldUseChineseSegmentation(text: text, language: language) {
            let chineseGranularity = resolveChineseGranularity(language: language, requested: granularity)
            return splitChinese(text, granularity: chineseGranularity)
        }
        return splitWhitespace(text)
    }

    /// Split on whitespace and punctuation boundaries (English, European languages)
    private static func splitWhitespace(_ text: String) -> [String] {
        // Split on whitespace, filter empty
        let raw = text.components(separatedBy: .whitespaces)
        return raw.filter { !$0.isEmpty }
    }

    private static func splitChinese(
        _ text: String,
        granularity: ForcedAlignmentGranularity
    ) -> [String] {
        var words: [String] = []
        var currentRun = ""
        var currentRunIsHan: Bool?

        func flushCurrentRun() {
            guard !currentRun.isEmpty else { return }
            defer {
                currentRun = ""
                currentRunIsHan = nil
            }

            if currentRunIsHan == true {
                switch granularity {
                case .word:
                    words.append(contentsOf: splitChineseWords(currentRun))
                case .automatic, .char:
                    words.append(contentsOf: currentRun.map { String($0) })
                }
            } else {
                let trimmed = currentRun.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    words.append(trimmed)
                }
            }
        }

        for character in text {
            if isDelimiter(character) {
                flushCurrentRun()
                continue
            }

            let isHan = character.unicodeScalars.allSatisfy(isHanScalar)
            if let currentRunIsHan, currentRunIsHan != isHan {
                flushCurrentRun()
            }

            currentRun.append(character)
            currentRunIsHan = isHan
        }

        flushCurrentRun()
        return words
    }

    private static func splitChineseWords(_ text: String) -> [String] {
        let nsText = text as NSString
        let locale = CFLocaleCreate(nil, "zh" as CFString)
        let tokenizer = CFStringTokenizerCreate(
            nil,
            nsText,
            CFRangeMake(0, nsText.length),
            kCFStringTokenizerUnitWord,
            locale
        )

        var words: [String] = []
        var result = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while result.rawValue != 0 {
            let range = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let token = nsText.substring(with: NSRange(location: range.location, length: range.length))
            let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty, !trimmed.allSatisfy(isDelimiter) {
                words.append(trimmed)
            }
            result = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }

        if words.isEmpty {
            return text.map { String($0) }
        }

        return words
    }

    private static func resolveChineseGranularity(
        language: String?,
        requested: ForcedAlignmentGranularity
    ) -> ForcedAlignmentGranularity {
        switch requested {
        case .word:
            return .word
        case .automatic, .char:
            return .char
        }
    }

    private static func shouldUseChineseSegmentation(text: String, language: String?) -> Bool {
        if let language, isChineseLanguage(language) {
            return true
        }

        let normalized = language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalized == nil || normalized == "auto" {
            return containsHan(text) && isMostlyUnspaced(text)
        }

        return false
    }

    private static func isChineseLanguage(_ language: String) -> Bool {
        let lang = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return lang == "zh" || lang.contains("chinese") || lang.contains("mandarin")
    }

    private static func containsHan(_ text: String) -> Bool {
        text.unicodeScalars.contains(where: isHanScalar)
    }

    private static func isMostlyUnspaced(_ text: String) -> Bool {
        let whitespaceCount = text.unicodeScalars.filter {
            CharacterSet.whitespacesAndNewlines.contains($0)
        }.count
        return whitespaceCount <= 1
    }

    private static func isDelimiter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy {
            CharacterSet.whitespacesAndNewlines.contains($0)
                || CharacterSet.punctuationCharacters.contains($0)
                || CharacterSet.symbols.contains($0)
        }
    }

    private static func isHanScalar(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        // CJK Unified Ideographs
        if v >= 0x4E00 && v <= 0x9FFF { return true }
        // CJK Extension A
        if v >= 0x3400 && v <= 0x4DBF { return true }
        // CJK Extension B+
        if v >= 0x20000 && v <= 0x2EBEF { return true }
        // CJK Compatibility Ideographs
        if v >= 0xF900 && v <= 0xFAFF { return true }
        return false
    }
}
