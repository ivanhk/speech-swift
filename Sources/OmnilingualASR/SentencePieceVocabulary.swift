import Foundation

/// Minimal SentencePiece `.model` reader for Omnilingual ASR.
///
/// Parses the protobuf wire format directly without requiring a protobuf
/// dependency — the model file is essentially a list of
/// `(piece: String, score: Float, type: Int32)` entries in field 1 of the
/// outer `ModelProto`.
///
/// The decoder mirrors fairseq2's `tokenizer.create_decoder(skip_special_tokens=True)`
/// behavior used by Meta's `ASRInferencePipeline`:
///
/// - Piece type `UNK`, `CONTROL`, `UNUSED`, or `BYTE` → stripped as special.
/// - Configured `pad`/`bos`/`eos`/`unk` ids → stripped as special.
/// - SentencePiece word-boundary marker `▁` (U+2581) → replaced with a space.
/// - Leading/trailing whitespace → trimmed.
public struct OmnilingualVocabulary: Sendable {
    /// Piece types as defined by sentencepiece_model.proto.
    private enum PieceType: Int32 {
        case normal = 1
        case unknown = 2
        case control = 3
        case userDefined = 4
        case byte = 6
        case unused = 5
    }

    private let pieces: [String]
    private let types: [PieceType]
    private let specialIds: Set<Int>

    public var count: Int { pieces.count }

    /// Load a SentencePiece `.model` file from disk.
    /// - Parameters:
    ///   - url: Path to `tokenizer.model`.
    ///   - tokenizer: Config section describing bos/pad/eos/unk ids.
    public static func load(
        from url: URL,
        tokenizer: OmnilingualConfig.Tokenizer
    ) throws -> OmnilingualVocabulary {
        let data = try Data(contentsOf: url)
        var pieces: [String] = []
        var types: [PieceType] = []

        var offset = 0
        while offset < data.count {
            let (fieldNumber, wireType, newOffset) = readTag(data: data, offset: offset)
            offset = newOffset

            guard fieldNumber == 1, wireType == 2 else {
                offset = skipField(data: data, offset: offset, wireType: wireType)
                continue
            }

            // Length-delimited SentencePiece submessage.
            let (length, dataOffset) = readVarint(data: data, offset: offset)
            offset = dataOffset
            let end = offset + length

            var piece: String = ""
            var pieceType: PieceType = .normal
            var subOffset = offset
            while subOffset < end {
                let (subField, subWire, subNewOffset) = readTag(data: data, offset: subOffset)
                subOffset = subNewOffset

                switch (subField, subWire) {
                case (1, 2):  // piece string
                    let (strLen, strOffset) = readVarint(data: data, offset: subOffset)
                    subOffset = strOffset
                    if let s = String(data: data[subOffset..<(subOffset + strLen)], encoding: .utf8) {
                        piece = s
                    }
                    subOffset += strLen
                case (3, 0):  // type varint
                    let (typeValue, nextOffset) = readVarint(data: data, offset: subOffset)
                    subOffset = nextOffset
                    pieceType = PieceType(rawValue: Int32(typeValue)) ?? .normal
                default:
                    subOffset = skipField(data: data, offset: subOffset, wireType: subWire)
                }
            }
            pieces.append(piece)
            types.append(pieceType)
            offset = end
        }

        guard !pieces.isEmpty else {
            throw OmnilingualVocabularyError.emptyVocabulary(url: url)
        }

        let specials: Set<Int> = [
            tokenizer.padIdx, tokenizer.bosIdx, tokenizer.eosIdx, tokenizer.unkIdx,
        ]
        return OmnilingualVocabulary(pieces: pieces, types: types, specialIds: specials)
    }

    private init(pieces: [String], types: [PieceType], specialIds: Set<Int>) {
        self.pieces = pieces
        self.types = types
        self.specialIds = specialIds
    }

    /// Decode token ids to text using fairseq2-style special-token skipping.
    /// Matches `self.tokenizer.create_decoder(skip_special_tokens=True)` in
    /// Meta's `ASRInferencePipeline`.
    public func decode(_ ids: [Int]) -> String {
        var result = ""
        for id in ids {
            guard id >= 0, id < pieces.count else { continue }
            if isSpecial(id) { continue }
            result += pieces[id]
        }
        return result
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    public func isSpecial(_ id: Int) -> Bool {
        if specialIds.contains(id) { return true }
        guard id >= 0, id < types.count else { return false }
        let type = types[id]
        return type == .control || type == .unknown || type == .unused || type == .byte
    }

    // MARK: - Protobuf wire format helpers

    private static func readVarint(data: Data, offset: Int) -> (value: Int, newOffset: Int) {
        var result = 0
        var shift = 0
        var off = offset
        while off < data.count {
            let byte = Int(data[off])
            off += 1
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0 { break }
            shift += 7
        }
        return (result, off)
    }

    private static func readTag(data: Data, offset: Int) -> (fieldNumber: Int, wireType: Int, newOffset: Int) {
        let (tag, newOffset) = readVarint(data: data, offset: offset)
        return (tag >> 3, tag & 0x07, newOffset)
    }

    private static func skipField(data: Data, offset: Int, wireType: Int) -> Int {
        switch wireType {
        case 0:
            let (_, newOffset) = readVarint(data: data, offset: offset)
            return newOffset
        case 1:
            return offset + 8
        case 2:
            let (length, newOffset) = readVarint(data: data, offset: offset)
            return newOffset + length
        case 5:
            return offset + 4
        default:
            return data.count
        }
    }
}

public enum OmnilingualVocabularyError: Error, CustomStringConvertible {
    case emptyVocabulary(url: URL)

    public var description: String {
        switch self {
        case .emptyVocabulary(let url):
            return "SentencePiece model at \(url.path) contained no pieces"
        }
    }
}
