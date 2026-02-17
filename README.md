# GreedyPhrase

A greedy phrase-based tokenizer that outperforms GPT-4 and GPT-4o tokenizers on compression, with a smaller vocabulary.

## Benchmark (enwik9, 1 GB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **222,805,405** | **4.49x** | **47 MB/s** |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x | 4.35 MB/s |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x | 7.13 MB/s |

GreedyPhrase achieves **1.23x better compression** than GPT-4 and **1.21x better** than GPT-4o, with a **1.5-3x smaller vocabulary** and **6-11x higher encoding throughput**.

## How It Works

1. **Phrase Mining** — Split text into atoms (words, punctuation, whitespace), then mine bigrams and trigrams. Top phrases fill 95% of vocabulary slots.
2. **BPE Fallback** — Train BPE on residual byte sequences not covered by phrases. BPE tokens fill the remaining 5% of vocabulary.
3. **Greedy Encoding** — Longest-match-first via a Trie. Falls back to byte-level tokens for unknown sequences (zero OOV errors).

The C backend (`fast_counter` + `fast_encoder`) handles gigabyte-scale datasets. `fast_counter` uses 12-thread parallel hashing with xxHash; `fast_encoder` uses mmap + contiguous trie pool. Full train-and-encode on enwik9 (1GB) completes in ~75 seconds.

## Quick Start

```bash
# Compile C backends (requires xxhash.h in project root)
gcc -O3 -march=native -pthread -o fast_counter fast_counter.c
gcc -O3 -march=native -o fast_encoder fast_encoder.c

# Train tokenizer on your corpus
python -c "
from greedyphrase import GreedyPhraseTokenizer
t = GreedyPhraseTokenizer(vocab_size=65536, model_path='tokenizer/greedyphrase.vocab')
t.train(['data/enwik9'], phrase_ratio=0.5)
"

# Encode a file
python -c "
from greedyphrase import GreedyPhraseTokenizer
t = GreedyPhraseTokenizer(vocab_size=65536, model_path='tokenizer/greedyphrase.vocab')
t.encode_file('data/enwik9', 'data/enwik9.tokens')
"
```

## Run Benchmarks

```bash
# GreedyPhrase on enwik9
python benchmark_enwik9.py

# Tiktoken (GPT-4, GPT-4o) on enwik9
pip install tiktoken
python benchmark/bench_tiktoken.py
```

## Project Structure

```
greedyphrase.py        # GreedyPhraseTokenizer — train, encode, decode
fast_counter.c         # Multithreaded n-gram counter (xxHash + 12 threads)
fast_encoder.c         # Trie-based greedy encoder (mmap + node pool + prefetch)
xxhash.h               # xxHash single-header library (vendored)
benchmark_enwik9.py    # GreedyPhrase enwik9 benchmark
benchmark/             # Baseline benchmarks (tiktoken, etc.)
tokenizer/             # Trained vocab files
data/                  # Datasets (enwik9)
GreedyPhrase.md        # Paper
RESEARCHPLAN.md        # Research plan and roadmap
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
