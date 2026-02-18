# GreedyPhrase

A greedy phrase-based tokenizer that outperforms GPT-4 and GPT-4o tokenizers on compression, with a smaller vocabulary.

## Benchmarks

### WikiText-103-raw (539 MB, clean Wikipedia prose)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **93,466,213** | **5.77x** | **39.6 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 120,196,189 | 4.49x | 11.9 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 119,160,774 | 4.53x | 7.1 MB/s |

**28% better compression** than tiktoken with **1/3 the vocab** and **3-6x faster encoding**.

### TinyStories (100 MB, natural English prose)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **11,237,250** | **8.90x** | **33.4 MB/s** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 24,541,816 | 4.07x | 10.9 MB/s |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 24,367,822 | 4.10x | 6.9 MB/s |

**2.2x better compression** than tiktoken — phrase-based tokenization excels on repetitive natural prose.

## How It Works

GreedyPhrase uses a **two-pass compound training** approach:

1. **Phrase Mining** — Split text into atoms (words, punctuation, whitespace), then count n-grams up to 7 atoms long. Top ~52K phrases become the primitive vocabulary.
2. **Compound Phrases** — Encode the corpus with the primitive vocab, then count consecutive token pairs. The top ~10K bigrams (each concatenating two phrases into a compound up to 14 atoms long) are added to the vocabulary.
3. **BPE Fallback** — Re-encode with the expanded vocab. Train BPE on residual byte sequences. ~3K BPE tokens fill the remaining slots.
4. **Greedy Encoding** — Longest-match-first via a Trie. Falls back to byte-level tokens for unknown sequences (zero OOV errors).

This two-pass approach captures long phrases (up to 14 atoms) without the memory cost of directly counting 14-grams.

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
