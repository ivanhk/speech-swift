import AppKit
import Foundation
import Observation
import ParakeetStreamingASR
import SpeechVAD

private let logPath = "/tmp/dictate.log"
private let logLock = NSLock()
private func dlog(_ msg: String) {
    logLock.lock()
    defer { logLock.unlock() }
    let line = "\(msg)\n"
    if let data = line.data(using: .utf8) {
        if let fh = FileHandle(forWritingAtPath: logPath) {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.closeFile()
        } else {
            FileManager.default.createFile(atPath: logPath, contents: data)
        }
    }
}

/// Handles audio buffering, VAD gating, and ASR processing off the main thread.
final class ASRProcessor: Sendable {
    private let session: StreamingSession
    private let vad: SileroVADModel
    private let lock = NSLock()
    private let _buffer = UnsafeMutablePointer<[Float]>.allocate(capacity: 1)
    nonisolated(unsafe) var lastSpeechProb: Float = 0

    private let speechThreshold: Float = 0.5
    private let silenceGateChunks = 15
    nonisolated(unsafe) private var silenceCounter = 0
    nonisolated(unsafe) private var speechActive = false

    init(session: StreamingSession, vad: SileroVADModel) {
        self.session = session
        self.vad = vad
        _buffer.initialize(to: [])
    }

    deinit {
        _buffer.deinitialize(count: 1); _buffer.deallocate()
    }

    var isSpeechActive: Bool { speechActive }

    /// Called from audio thread — buffers ALL audio (speech + silence).
    func appendAudio(_ samples: [Float]) {
        lock.lock()
        _buffer.pointee.append(contentsOf: samples)
        lock.unlock()
    }

    /// Run VAD on samples (called from processQueue, not audio thread).
    func runVAD(on samples: [Float]) {
        var offset = 0
        while offset + 512 <= samples.count {
            let vadChunk = Array(samples[offset..<offset + 512])
            let prob = vad.processChunk(vadChunk)
            lastSpeechProb = prob
            if prob >= speechThreshold {
                speechActive = true
                silenceCounter = 0
            } else {
                silenceCounter += 1
                if silenceCounter >= silenceGateChunks {
                    speechActive = false
                }
            }
            offset += 512
        }
    }

    var bufferedCount: Int {
        lock.lock()
        let c = _buffer.pointee.count
        lock.unlock()
        return c
    }

    /// Called from processQueue — runs VAD + ASR.
    func processBuffered() -> (partials: [ParakeetStreamingASRModel.PartialTranscript], speaking: Bool) {
        lock.lock()
        let chunk = _buffer.pointee
        _buffer.pointee.removeAll(keepingCapacity: true)
        lock.unlock()

        guard !chunk.isEmpty else { return ([], speechActive) }

        // Run VAD for UI indicator (on processQueue, not audio thread)
        runVAD(on: chunk)

        // Always feed audio to encoder (speech + silence).
        // The cache-aware encoder needs continuous input for context.
        // VAD is used only for UI indicator, not gating.
        do {
            let partials = try session.pushAudio(chunk)
            if !partials.isEmpty {
                dlog("ASR got \(partials.count) partials: \(partials.map { $0.text })")
            }
            return (partials, speechActive)
        } catch {
            dlog("ASR error: \(error)")
            return ([], speechActive)
        }
    }

    func finalize() -> [ParakeetStreamingASRModel.PartialTranscript] {
        let (remaining, _) = processBuffered()
        do {
            return remaining + (try session.finalize())
        } catch {
            return remaining
        }
    }
}

@Observable
@MainActor
final class DictateViewModel {
    var sentences: [String] = []
    var partialText = ""
    var isRecording = false
    var isLoading = false
    var loadingStatus = ""
    var errorMessage: String?
    var isSpeechActive = false
    var recordingStartTime: Date?

    private var model: ParakeetStreamingASRModel?
    private var vad: SileroVADModel?
    private var processor: ASRProcessor?
    private let recorder = StreamingRecorder()
    private let processQueue = DispatchQueue(label: "dictate.asr", qos: .userInteractive)
    private var processTimer: DispatchSourceTimer?

    var modelLoaded: Bool { model != nil && vad != nil }
    var audioLevel: Float { recorder.audioLevel }

    init() {
        Task { await loadModels() }
    }

    var wordCount: Int {
        let all = sentences.joined(separator: " ") + (partialText.isEmpty ? "" : " " + partialText)
        return all.split(separator: " ").count
    }

    var fullText: String {
        let committed = sentences.joined(separator: "\n")
        if committed.isEmpty { return partialText }
        if partialText.isEmpty { return committed }
        return committed + "\n" + partialText
    }

    // MARK: - Model Loading

    func loadModels() async {
        guard model == nil else { return }
        isLoading = true
        errorMessage = nil

        do {
            loadingStatus = "Downloading ASR model..."
            let loaded = try await Task.detached {
                try await ParakeetStreamingASRModel.fromPretrained { [weak self] progress, status in
                    DispatchQueue.main.async {
                        self?.loadingStatus = status.isEmpty
                            ? "Downloading ASR... \(Int(progress * 100))%"
                            : "\(status) (\(Int(progress * 100))%)"
                    }
                }
            }.value

            loadingStatus = "Warming up ASR..."
            try loaded.warmUp()
            model = loaded

            loadingStatus = "Loading VAD..."
            let vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml)
            }.value
            vad = vadModel
            loadingStatus = ""
            dlog("Models loaded (ASR + VAD)")
        } catch {
            errorMessage = "Failed to load: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    // MARK: - Recording

    func toggleRecording() {
        if isRecording { stopRecording() } else { startRecording() }
    }

    func startRecording() {
        guard let model, let vad else { return }
        errorMessage = nil
        partialText = ""
        sentences.removeAll()
        recordingStartTime = Date()

        do {
            let session = try model.createSession()
            let proc = ASRProcessor(session: session, vad: vad)
            processor = proc

            // Audio callback — just buffers, no inference
            recorder.start { [proc] chunk in
                proc.appendAudio(chunk)
            }

            // DispatchSourceTimer drains buffer on processQueue every 300ms
            let timer = DispatchSource.makeTimerSource(queue: processQueue)
            timer.schedule(deadline: .now(), repeating: .milliseconds(300))
            timer.setEventHandler { [weak self, proc] in
                let buffered = proc.bufferedCount
                if buffered > 0 { dlog("Timer: \(buffered) buffered") }
                let (partials, speaking) = proc.processBuffered()
                Task { @MainActor [weak self] in
                    self?.isSpeechActive = speaking
                    for partial in partials {
                        if partial.isFinal && !partial.text.isEmpty {
                            dlog("UI commit: '\(partial.text)'")
                            self?.sentences.append(partial.text)
                            self?.partialText = ""
                        } else if !partial.text.isEmpty {
                            self?.partialText = partial.text
                        }
                    }
                }
            }
            timer.resume()
            processTimer = timer

            isRecording = true
            dlog("Recording started")
        } catch {
            errorMessage = "Failed to start: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        processTimer?.cancel()
        processTimer = nil
        recorder.stop()
        isRecording = false
        isSpeechActive = false
        recordingStartTime = nil

        guard let processor else { return }
        let finals = processor.finalize()
        for partial in finals {
            if !partial.text.isEmpty {
                sentences.append(partial.text)
            }
        }
        self.processor = nil
        partialText = ""
    }

    // MARK: - Actions

    func pasteToFrontApp() {
        let text = fullText
        guard !text.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let src = CGEventSource(stateID: .hidSystemState)
            let keyDown = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: true)
            keyDown?.flags = .maskCommand
            let keyUp = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: false)
            keyUp?.flags = .maskCommand
            keyDown?.post(tap: .cghidEventTap)
            keyUp?.post(tap: .cghidEventTap)
        }
    }

    func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
    }

    func clearText() {
        sentences.removeAll()
        partialText = ""
    }
}
