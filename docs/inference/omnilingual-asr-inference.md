# Omnilingual ASR — Inference Pipeline

Inference path for Meta's Omnilingual ASR (CTC variant) on Apple Silicon.
For architecture and weights, see [`docs/models/omnilingual-asr.md`](../models/omnilingual-asr.md).

## Quick start

```swift
import OmnilingualASR
import AudioCommon

let model = try await OmnilingualASRModel.fromPretrained()
let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
let text = try model.transcribeAudio(audio, sampleRate: 16000)
print(text)
```

The default model is the 10 s CoreML INT8 variant
(`aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s`). For lower memory
and faster cold start, use the 5 s variant:

```swift
let model = try await OmnilingualASRModel.fromPretrained(
    modelId: OmnilingualASRModel.shortWindowModelId)
```

## CLI

```bash
audio transcribe recording.wav --engine omnilingual           # 10 s window (default)
audio transcribe recording.wav --engine omnilingual --window 5  # 5 s window
```

## Pipeline

The Swift implementation mirrors Meta's `ASRInferencePipeline.transcribe()`:

```
1. Resample → 16 kHz mono Float32  (AudioFileLoader.resample)
2. Hard cap at 40 s                (matches MAX_ALLOWED_AUDIO_SEC)
3. Chunk into N × inputSamples windows (10 s or 5 s)
4. layer_norm raw waveform per chunk: (x - mean) / sqrt(var + 1e-5)
5. Zero-pad chunk to inputSamples
6. CoreML forward → [1, T, 10288] logits
7. Greedy CTC: argmax → collapse consecutive duplicates
8. SentencePiece detokenize, skip_special_tokens=True
```

### Why per-chunk layer_norm

The reference pipeline normalizes the raw waveform of the **whole utterance**
before any chunking. Because the CoreML graph is traced at a fixed window,
this Swift port chunks first and normalizes each chunk's real content
(before zero padding), so the mean/variance stats are computed over actual
audio rather than silence. For sub-window inputs this matches reference
behavior; for multi-window inputs each chunk is independently normalized
(a small divergence from reference batch behavior, acceptable for utterances
that already fit in one window — the typical case).

### 40 s hard cap

Meta's pipeline enforces `MAX_ALLOWED_AUDIO_SEC = 40` because the encoder
was trained on ≤30 s clips and longer inputs degrade quality / use too much
memory. This Swift module enforces the same cap at the API boundary:

```swift
do {
    _ = try model.transcribeAudio(longAudio, sampleRate: 16000)
} catch {
    // "Input 45.2s exceeds Omnilingual cap of 40s. Segment with SpeechVAD or use ParakeetStreamingASR."
}
```

For longer audio, segment first with `SpeechVAD` (or for streaming dictation,
use `ParakeetStreamingASR` which is built around endpoint detection).

## Multilingual support

The CTC variant covers Meta's full 1600+ language catalog through a single
shared 10288-entry SentencePiece vocabulary. **No language hint is required**
— the `language` parameter on `transcribe(audio:sampleRate:language:)` is
accepted for protocol conformance but ignored by the CTC head. (For
language-conditioned inference, use the LLM variant when ported in a
follow-up module.)

The repo includes FLEURS test fixtures for English, Hindi, French, and
Arabic in `Tests/OmnilingualASRTests/Resources/fleurs_*.wav`. Each E2E test
verifies the model produces script-correct output (Latin / Devanagari /
Arabic) and at least one expected content word.

## Performance

CoreML INT8, M3 Pro, 10 s window:
- Cold start (download + compile): ~30 s
- Warmup: ~1.5 s
- Per-window inference: ~120 ms (RTF ≈ 0.012 — ~80× realtime)
- Memory footprint: ~312 MB on disk; ~600 MB resident with CoreML overhead

## See also

- [docs/models/omnilingual-asr.md](../models/omnilingual-asr.md) — model card, architecture, variants
- [Reference Python pipeline](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/pipeline.py)
- [Omnilingual ASR paper](https://arxiv.org/abs/2511.09690)
