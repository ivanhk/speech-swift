# Forced Aligner ([Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B))

## Overview

Qwen3-ForcedAligner predicts timestamps for audio+text pairs. It shares the same encoder-decoder architecture as Qwen3-ASR but replaces the vocabulary lm_head with a 5000-class timestamp classification head. Inference is non-autoregressive (single forward pass through the decoder).

```
Audio (16kHz) + Text
    |            |
    v            v
+------------------+   +---------------------+
|  Mel → Audio     |   |  Word splitting     |
|  Encoder (24L)   |   |  + timestamp slots  |
+--------+---------+   +---------+-----------+
         |                        |
         v                        v
+------------------------------------------------+
|  Text Decoder (28L, single forward pass)       |
|  Audio embeds injected at <audio_pad> positions |
|  Timestamp tokens at word boundaries            |
+-----------------------+------------------------+
                        |
                        v
+------------------------------------------------+
|  Classify Head (Linear 1024 → 5000)            |
|  argmax at <timestamp> positions                |
+-----------------------+------------------------+
                        |
                        v
+------------------------------------------------+
|  LIS Monotonicity Correction                   |
|  Index × 80ms → timestamps in seconds          |
+-----------------------+------------------------+
                        |
                        v
               [AlignedWord] array
        (alignment unit, startTime, endTime)
```

## Architecture

| Component | Config |
|-----------|--------|
| Audio encoder | 24 layers, d_model=1024, 16 heads, FFN=4096, output→1024 |
| Text decoder | 28 layers, hidden=1024, 16Q/8KV heads, headDim=128 (4-bit, 8-bit, or bf16) |
| Classify head | Linear(1024, 5000), float16 (NOT tied to embeddings) |
| Timestamp resolution | 80ms per class (5000 classes = 400s max) |

## Key Difference from ASR

| | ASR | Forced Aligner |
|---|-----|----------------|
| Decoder mode | Autoregressive (token by token) | Non-autoregressive (single pass) |
| Output head | Tied embedding lm_head (vocab 151936) | Classify head (5000 timestamp classes) |
| KV cache | Yes (grows with each token) | None |
| Input | Audio only | Audio + text with `<timestamp>` slots |
| Audio encoder | 18L/896D (0.6B) | 24L/1024D (larger) |

## Inference Pipeline

### 1. Audio Encoding
Same as ASR: mel spectrogram → chunked Conv2D → transformer → projector.

### 2. Text Preprocessing (TextPreprocessing.swift)

Text is split into alignment units (language-specific) and `<timestamp>` tokens inserted:

**English:** Split on whitespace
```
"Can you guarantee" → ["Can", "you", "guarantee"]
```

**Chinese / Han text, default:** Character-level splitting
```
"你好世界" → ["你", "好", "世", "界"]
```

**Chinese / Han text, `--word-level`:** Word segmentation via `CFStringTokenizer`
```
"你好世界" → ["你好", "世界"]
```

Each alignment unit gets `<timestamp>` pairs:
```
<ts>Can<ts> <ts>you<ts> <ts>guarantee<ts>
```

### Granularity Controls

`audio align` keeps Chinese-compatible defaults:

- Default Chinese/Han behavior is `char-level`
- `--word-level` opts into tokenizer-based Chinese word segmentation
- `--char-level` is available for explicitness and for scripts that mix Latin and Han text
- These flags only affect Chinese/Han text; whitespace-delimited languages keep the existing behavior
- `--sentence-level` switches the CLI to sentence-by-sentence alignment on the remaining audio, returning one final range per sentence
- `--sentence-level` applies to all languages and can be combined with `--char-level` or `--word-level`

Language handling:

- `--language zh` / `--language chinese` forces Chinese segmentation rules
- `--language auto` is treated the same as omitting `--language` entirely
- With no language hint, the aligner auto-detects Han text heuristically and still defaults to `char-level`

### 3. Single Forward Pass

Build the full sequence with chat template:
```
<|im_start|>system\n<|im_end|>\n
<|im_start|>user\n<|audio_start|>[audio_pad × N]<|audio_end|><|im_end|>\n
<|im_start|>assistant\n
<ts>word1_tokens<ts> <ts>word2_tokens<ts> ...
```

One forward pass through the decoder (no cache, no loop). Apply classify head to all hidden states → logits `[1, seqLen, 5000]`.

### 4. Timestamp Extraction

1. Extract logits only at `<timestamp>` positions
2. argmax → raw timestamp class indices
3. Multiply by 80ms → raw timestamps in seconds
4. Pair consecutive timestamps as (start, end) per alignment unit

### 5. Optional Sentence-Level CLI Alignment

When `audio align --sentence-level` is used, the CLI performs one extra post-processing step:

1. Split the original text into sentences with Foundation `enumerateSubstrings(..., .bySentences)`
2. Align the first sentence against the full audio
3. Advance to the end timestamp of that sentence
4. Align the next sentence against the remaining audio, and repeat

This keeps sentence punctuation in the final output and improves long-text stability because each forced-alignment pass sees only one sentence of text instead of the entire transcript.

### 6. LIS Monotonicity Correction (TimestampCorrection.swift)

Raw timestamps may not be monotonic. Fix via:
1. Find Longest Increasing Subsequence (O(n log n))
2. Small gaps (≤2 positions): nearest-neighbor correction
3. Larger gaps: linear interpolation between LIS anchors
4. Final pass: enforce non-decreasing order

## Performance (M2 Max, 64 GB)

| Stage | Time | Notes |
|-------|------|-------|
| Audio encoder | ~328ms | Mel extraction + 24L transformer + projector |
| Decoder + classify | ~37ms | Single forward pass, no autoregressive loop |
| **Total (20s audio)** | **~365ms** | **RTF ~0.018 (55x faster than real-time)** |

Debug build. Release would be faster.

## Weight Structure

Weights use a `thinker.` prefix:

| Key pattern | Component |
|-------------|-----------|
| `thinker.audio_tower.*` | Audio encoder (float16) |
| `thinker.model.*` | Text decoder (4-bit quantized) |
| `thinker.lm_head.weight` | Classify head (float16, NOT quantized) |

## Model Files

| Model | ID | Size |
|-------|----|------|
| MLX 4-bit | `aufklarer/Qwen3-ForcedAligner-0.6B-4bit` | ~979 MB |
| MLX 8-bit | `aufklarer/Qwen3-ForcedAligner-0.6B-8bit` | ~1.3 GB |
| MLX bf16 | `aufklarer/Qwen3-ForcedAligner-0.6B-bf16` | ~1.8 GB |
| CoreML INT4 | `aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4` | ~662 MB |
| CoreML INT8 | `aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8` | ~1.1 GB |

Variant is auto-detected from `quantize_config.json` in the model directory. The bf16 variant uses a float text decoder (`FloatTextModel`) instead of a quantized one.

## CLI Usage

```bash
# Align with provided text
audio align audio.wav --text "Can you guarantee that the replacement part will be shipped tomorrow?"

# Transcribe first, then align
audio align audio.wav

# Chinese defaults to character-level alignment
audio align zh.wav --text "一九零八年的春天" --language zh

# Chinese word-level alignment
audio align zh.wav --text "一九零八年的春天" --language zh --word-level

# Sentence-level output (all languages)
audio align audio.wav --text "Hello world. This is a test." --sentence-level

# Chinese sentence-level output with word-level internal segmentation
audio align zh.wav --text "一九零八年的春天。在奥地利一个小镇上。" --language zh --word-level --sentence-level

# Explicit auto language (same as omitting --language)
audio align zh.wav --language auto --word-level

# Custom aligner model (8-bit or bf16)
audio align audio.wav --aligner-model aufklarer/Qwen3-ForcedAligner-0.6B-8bit
audio align audio.wav --aligner-model aufklarer/Qwen3-ForcedAligner-0.6B-bf16
```

Output format:
```
[0.12s - 0.45s] Can
[0.45s - 0.72s] you
[0.72s - 1.20s] guarantee
...
```

Sentence-level output:
```
[0.12s - 1.20s] Can you guarantee?
[1.20s - 2.40s] The replacement part will be shipped tomorrow.
```

## Swift API

```swift
let aligner = try await Qwen3ForcedAligner.fromPretrained()

let aligned = aligner.align(
    audio: audioSamples,
    text: "Can you guarantee that the replacement part will be shipped tomorrow?",
    sampleRate: 24000
)

for word in aligned {
    print("[\(word.startTime)s - \(word.endTime)s] \(word.text)")
}
```

## Conversion

```bash
# MLX variants
python scripts/convert_forced_aligner.py --bits 4 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-4bit
python scripts/convert_forced_aligner.py --bits 8 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-8bit
python scripts/convert_forced_aligner.py --bits 0 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-bf16

# CoreML variants
python scripts/convert_forced_aligner.py --coreml --coreml-bits 4 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT4
python scripts/convert_forced_aligner.py --coreml --coreml-bits 8 --upload --repo-id aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8
```

MLX: quantizes text decoder (attention + MLP + embeddings) to N-bit. Audio encoder and classify head kept as float16. `--bits 0` keeps everything as float16.

CoreML: traces audio encoder, text decoder + classify head, and embedding as separate models with INT4/INT8 palettization. Published as pre-compiled `.mlmodelc` bundles.
