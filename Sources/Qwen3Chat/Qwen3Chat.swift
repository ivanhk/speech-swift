import CoreML
import Foundation
import AudioCommon

/// Backward-compatible facade for the current CoreML chat runtime.
///
/// `Qwen3ChatModel` historically pointed at an older CoreML implementation that
/// no longer exists in the repository. Keep the public surface stable by
/// delegating to `Qwen35CoreMLChat`, which is the maintained CoreML backend.
public final class Qwen3ChatModel: @unchecked Sendable {
    public static let defaultModelId = Qwen35CoreMLChat.defaultModelId

    private let backend: Qwen35CoreMLChat
    private var conversationHistory: [ChatMessage] = []
    private var cachedSystemPrompt: String?

    /// Quantization option retained for source compatibility.
    ///
    /// The maintained CoreML backend currently only ships INT8 bundles, so both
    /// cases map to the same runtime variant.
    public enum Quantization: String, Sendable {
        case int4
        case int8

        var backendValue: Qwen35CoreMLChat.Quantization { .int8 }
    }

    public var config: Qwen3ChatConfig { backend.config }
    public var tokenizer: ChatTokenizer { backend.tokenizer }
    public var lastMetrics: (tokensPerSec: Double, prefillMs: Double, decodeMs: Double, msPerToken: Double) {
        backend.lastMetrics
    }

    private init(backend: Qwen35CoreMLChat) {
        self.backend = backend
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        quantization: Quantization = .int8,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ChatModel {
        let backend = try await Qwen35CoreMLChat.fromPretrained(
            modelId: modelId,
            quantization: quantization.backendValue,
            computeUnits: computeUnits,
            progressHandler: progressHandler
        )
        return Qwen3ChatModel(backend: backend)
    }

    public static func fromLocal(
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> Qwen3ChatModel {
        let backend = try await Qwen35CoreMLChat.fromLocal(
            directory: directory,
            computeUnits: computeUnits
        )
        return Qwen3ChatModel(backend: backend)
    }

    public func generate(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        try backend.generate(messages: messages, sampling: sampling)
    }

    public func generateStream(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        backend.generateStream(messages: messages, sampling: sampling)
    }

    public func chat(
        _ userMessage: String,
        systemPrompt: String? = nil,
        sampling: ChatSamplingConfig = .default
    ) throws -> String {
        let messages = buildMessages(userMessage: userMessage, systemPrompt: systemPrompt)
        let response = try backend.generate(messages: messages, sampling: sampling)
        appendTurn(userMessage: userMessage, assistantMessage: response, systemPrompt: systemPrompt)
        return response
    }

    public func chatStream(
        _ userMessage: String,
        systemPrompt: String? = nil,
        sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        let messages = buildMessages(userMessage: userMessage, systemPrompt: systemPrompt)

        return AsyncThrowingStream { continuation in
            Task {
                var collected = ""
                do {
                    for try await token in backend.generateStream(messages: messages, sampling: sampling) {
                        collected += token
                        continuation.yield(token)
                    }
                    appendTurn(userMessage: userMessage, assistantMessage: collected, systemPrompt: systemPrompt)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func resetConversation() {
        conversationHistory = []
        cachedSystemPrompt = nil
        backend.resetState()
    }

    private func buildMessages(userMessage: String, systemPrompt: String?) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        let effectiveSystem = systemPrompt ?? cachedSystemPrompt
        if let effectiveSystem, !effectiveSystem.isEmpty {
            messages.append(ChatMessage(role: .system, content: effectiveSystem))
        }
        messages.append(contentsOf: conversationHistory)
        messages.append(ChatMessage(role: .user, content: userMessage))
        return messages
    }

    private func appendTurn(userMessage: String, assistantMessage: String, systemPrompt: String?) {
        if let systemPrompt, !systemPrompt.isEmpty {
            cachedSystemPrompt = systemPrompt
        }
        conversationHistory.append(ChatMessage(role: .user, content: userMessage))
        conversationHistory.append(ChatMessage(role: .assistant, content: assistantMessage))
    }
}
