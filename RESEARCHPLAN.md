# GreedyPhrase Research Plan

## Phase 1: Foundation (Complete)
- **Goal:** Validate "Compression is Intelligence" hypothesis.
- **Status:** **Success.**
    - Implemented `GreedyPhraseTokenizer` (Greedy Phrase-Based).
    - Achieved **4.39x** compression ratio on enwik9 (vs GPT-4 3.65x, GPT-4o 3.70x).
    - Built initial C backends (`fast_counter`, `fast_encoder`).
    - Fixed path bug (`counts.txt` -> `tokenizer/counts.txt`) and encoding bug (UTF-8 -> latin-1 for non-ASCII corpus data).

## Phase 2: Hybrid Tokenization (Complete)
- **Goal:** Replace raw-byte fallback with BPE to improve compression on residuals.
- **Status:** **Success.**
    - Implemented 95% phrase / 5% BPE vocabulary split.
    - First-pass encode with phrase vocab collects residual byte-fallback runs.
    - BPE trained on residuals fills remaining vocabulary slots.
    - Compression improved from **4.39x → 4.49x** on enwik9.

## Phase 3: C Backend Optimization (Complete)
- **Goal:** Make the pipeline fast enough for rapid iteration on 1GB+ datasets.
- **Status:** **Success.**
    - `fast_encoder`: mmap + contiguous trie pool + buffered output + prefetch → **47 MB/s** (was 0.37 MB/s, 127x speedup).
    - `fast_counter`: 12-thread parallel counting + xxHash + arena allocation + 128M-bucket merge tables → **81 MB/s** counting, 29s total (was ~160s single-threaded).
    - Full train-and-encode pipeline on enwik9: **~75 seconds** (was 45+ minutes for encoding alone).

### Phase 3 Benchmark — enwik9 (1 GB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **222,805,405** | **4.49x** | **47 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x | 7.13 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x | 4.35 MB/s |

## Phase 4: Two-Pass Compound Training (Complete)
- **Goal:** Capture long phrases (up to 14 atoms) without OOM on large corpora.
- **Status:** **Success.**
    - Bumped MAX_NGRAM from 6 to 7 in fast_counter.
    - Added two-pass compound training: Pass 1 encodes with ~52K primitive phrases, counts token bigrams, selects ~10K compound phrases, Pass 2 re-encodes with expanded vocab.
    - Token bigram counting uses numpy vectorized `(a << 16) | b` uint32 packing + `np.unique`.
    - Vocab budget: 52K primitive + 10K compound + 3.2K BPE = 65.3K.

### Phase 4 Benchmarks

#### WikiText-103-raw (539 MB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Enc MB/s |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **93,466,213** | **5.77x** | **39.6** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 120,196,189 | 4.49x | 11.9 |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 119,160,774 | 4.53x | 7.1 |

#### TinyStories (100 MB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Enc MB/s |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **11,237,250** | **8.90x** | **33.4** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 24,541,816 | 4.07x | 10.9 |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 24,367,822 | 4.10x | 6.9 |

## Phase 5: Next Steps
- **Multi-pass compounding:** Could a third pass (bigrams of compound tokens) push even further?
- **Corpus-adaptive budget:** Auto-tune primitive/compound/BPE ratio based on corpus statistics.
- **Model training:** Train GreedyPhrase-1B to validate the "compression is intelligence" hypothesis end-to-end.
