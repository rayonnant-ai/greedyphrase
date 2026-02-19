# GreedyPhrase

A greedy phrase-based tokenizer that outperforms GPT-4 and GPT-4o tokenizers on compression, with a smaller vocabulary.

## Benchmarks

### WikiText-103-raw (539 MB, clean Wikipedia prose)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **89,291,627** | **6.04x** | **42.5 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 120,196,189 | 4.49x | 11.9 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 119,160,774 | 4.53x | 7.1 MB/s |

**34% better compression** than tiktoken with **1/3 the vocab** and **3-6x faster encoding**.

### TinyStories (100 MB, natural English prose)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **10,890,713** | **9.18x** | **36.9 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 24,541,816 | 4.07x | 10.9 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 24,367,822 | 4.10x | 6.9 MB/s |

**2.24x better compression** than tiktoken — phrase-based tokenization excels on repetitive natural prose.

### enwik9 (1 GB, heavy XML markup)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (No Templates)** | **65,536** | **174,782,848** | **5.72x** | **43.6 MB/s** |
| **GreedyPhrase (Templates)** | **65,536** | **175,675,079** | **5.69x** | **40.7 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x | 7.13 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x | 4.35 MB/s |

GreedyPhrase outperforms GPT-4's tokenizer by **1.57x** and GPT-4o's by **1.55x** on enwik9, while using a vocabulary **1.5–3x smaller**. The gap is especially notable given that enwik9's heavy XML markup and non-natural-language content penalizes phrase-based approaches. While linguistic templates were selected by the market (1,664 patterns), the non-templated version achieved slightly better raw compression, indicating that primitive/compound tokens remain the most efficient way to capture XML structure.

## How It Works

GreedyPhrase uses **iterative compound training** (3 passes by default):

1. **Linguistic Mining** — Scans raw text for structural templates (e.g., "certified ;? by the RIAA") by masking entities (numbers, names) and counting patterns. The literal words from top patterns are injected into the vocabulary pool.
2. **Phrase Mining** — Split text into atoms (words, punctuation, whitespace), then count n-grams up to 7 atoms long. Top ~52K phrases become the primitive vocabulary.
3. **Compound Pass 1** — Encode the corpus with the primitive vocab, then count consecutive token pairs. The top ~5K bigrams (each concatenating two phrases into a compound up to 14 atoms) are added to the vocabulary.
4. **Compound Pass 2** — Re-encode with the expanded vocab and count token pairs again. The top ~5K bigrams of compound tokens yield triple-compounds up to 21+ atoms long.
5. **BPE Fallback** — Re-encode with the full vocab. Train BPE on residual byte sequences. ~3K BPE tokens fill the remaining slots.
6. **Greedy Encoding** — Longest-match-first via a Trie. Falls back to byte-level tokens for unknown sequences (zero OOV errors).

Each compounding pass doubles the maximum phrase reach without ever counting high-order n-grams directly (which would OOM on large corpora).

## Configuration Options

You can tune the tokenizer by passing arguments to the `train()` method or the constructor:

- **`vocab_size`** (default: `65536`): The total number of token IDs available. GreedyPhrase uses a "Free Market" approach where all candidates (primitives, compounds, templates, BPE) compete for these slots based on frequency.
- **`template_budget`** (default: `2000`): The maximum number of linguistic templates to mine and promote. 
  - To **disable linguistic templating**, set `template_budget=0`.
- **`compound_passes`** (default: `2`): The number of iterative encoding passes to identify long multi-word phrases. Each pass doubles the effective reach. 
  - To **disable compounding**, set `compound_passes=0`.
- **`bpe_slots`** (default: `3000`): The number of slots reserved for Byte-Pair Encoding (BPE) fallback on residuals. This ensures robust coverage of unknown words.

### Example: Training with custom options

```python
from greedyphrase import GreedyPhraseTokenizer

t = GreedyPhraseTokenizer(vocab_size=32768)
t.train(
    ['data/my_corpus.txt'], 
    template_budget=500,   # Fewer templates
    compound_passes=3,     # deeper compounding
    bpe_slots=1000         # smaller BPE fallback
)
```

The C backend (`fast_counter` + `fast_encoder`) handles gigabyte-scale datasets. `fast_counter` uses 12-thread parallel hashing with xxHash; `fast_encoder` uses mmap + contiguous trie pool with speculative prefetch.

## Quick Start

```bash
# Compile C backends (requires xxhash.h in project root)
gcc -O3 -march=native -pthread -o fast_counter fast_counter.c
gcc -O3 -march=native -o fast_encoder fast_encoder.c

# Train tokenizer on your corpus
python -c "
from greedyphrase import GreedyPhraseTokenizer
t = GreedyPhraseTokenizer(vocab_size=65536, model_path='tokenizer/greedyphrase.vocab')
t.train(['data/wikitext103.txt'])
"

# Encode a file
python -c "
from greedyphrase import GreedyPhraseTokenizer
t = GreedyPhraseTokenizer(vocab_size=65536, model_path='tokenizer/greedyphrase.vocab')
t.encode_file('data/wikitext103.txt', 'data/wikitext103.tokens')
"
```

## Run Benchmarks

```bash
# WikiText-103
python benchmark_wikitext.py

# TinyStories
python benchmark_tinystories.py

# enwik9
python benchmark_enwik9.py
```

## Project Structure

```
greedyphrase.py           # GreedyPhraseTokenizer — train, encode, decode
fast_counter.c            # Multithreaded n-gram counter (xxHash, 7-gram, 12 threads)
fast_encoder.c            # Trie-based greedy encoder (mmap + node pool + prefetch)
xxhash.h                  # xxHash single-header library (vendored)
benchmark_wikitext.py     # WikiText-103 benchmark (GreedyPhrase + tiktoken)
benchmark_tinystories.py  # TinyStories benchmark
benchmark_enwik9.py       # enwik9 benchmark
tokenizer/                # Trained vocab files
data/                     # Datasets
GreedyPhrase.md           # Paper
RESEARCHPLAN.md           # Research plan and roadmap
RESEARCHLOG.md            # Detailed engineering log
```

## Citation

```bibtex
@misc{rezel2026greedyphrase,
  author = {Rezel, Rohana},
  title = {GreedyPhrase Tokenizer: Maximizing Effective Context via Greedy Phrase Compression},
  year = {2026},
  publisher = {Rayonnant AI},
  howpublished = {\url{https://github.com/rayonnant-ai/greedyphrase}},
  email = {rohana.rezel@rayonnant.ai}
}
```
