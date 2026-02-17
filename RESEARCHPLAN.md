# GreedyPhrase Research Plan

## Phase 1: Foundation (Complete)
- **Goal:** Validate "Compression is Intelligence" hypothesis.
- **Status:** **Success.**
    - Implemented `GreedyPhraseTokenizer` (Greedy Phrase-Based).
    - Achieved **4.39x** compression ratio on enwik9 (vs GPT-4 3.65x, GPT-4o 3.70x).
    - Optimized training pipeline with C backends (`fast_counter`, `fast_encoder`).
    - Fixed path bug (`counts.txt` -> `tokenizer/counts.txt`) and encoding bug (UTF-8 -> latin-1 for non-ASCII corpus data).

### Phase 1 Baseline — enwik9 Benchmark (Apples-to-Apples)

All tokenizers benchmarked on enwik9 (1.00 GB, raw Wikipedia XML):

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **227,952,298** | **4.39x** (bytes/token) | 0.39 MB/s |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x (bytes/token) | 7.13 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x (bytes/token) | 4.35 MB/s |

**Notes:** GreedyPhrase achieves **1.20x** better compression than GPT-4's tokenizer and **1.19x** better than GPT-4o's on enwik9, despite using a vocabulary **1.5–3x smaller**. Tiktoken throughput is higher due to its optimized Rust backend.

## Phase 2: Hybrid Tokenization (Current)
- **Problem:** The current fallback for non-phrases is raw bytes (compression 1.0x). This is especially costly on enwik9 where markup and rare tokens are abundant.
- **Hypothesis:** Using **BPE (Byte Pair Encoding)** as the fallback mechanism instead of bytes will significantly boost overall compression.
- **Proposed Architecture:**
    1.  **Layer 1:** Greedy Phrase Match (Top 50% vocab). Captures "superwords" (e.g., "guest appearance").
    2.  **Layer 2:** BPE Fallback (Remaining 50% vocab). Captures subwords (e.g., "ing", "tion", "un") for the residual text.
- **Expected Outcome:**
    - Boost compression from **4.39x** (enwik9 baseline, already 1.20x better than GPT-4) -> **~6.0x+** on enwik9.
    - Better handling of rare words, markup, and morphology.
- **Implementation Plan:**
    - Integrate a lightweight C++ BPE trainer (like `sentencepiece` or a custom C impl) into the pipeline.
    - Modify `fast_encoder.c` to support a two-stage lookup (Trie -> BPE).

