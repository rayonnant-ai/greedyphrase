# GreedyPhrase

A greedy phrase-based tokenizer that outperforms GPT-4 and GPT-4o tokenizers on compression, with a smaller vocabulary.

## Benchmark (enwik9, 1 GB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio |
| :--- | :--- | :--- | :--- |
| **GreedyPhrase** | **65,536** | **227,952,298** | **4.39x** |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x |

GreedyPhrase achieves **1.20x better compression** than GPT-4 and **1.19x better** than GPT-4o, with a **1.5-3x smaller vocabulary**.

## How It Works

1. **Atomic Tokenization** — Split text into atoms (words, punctuation, whitespace). Top frequent atoms fill 50% of the vocabulary.
2. **Phrase Mining** — Mine the corpus for frequent bigrams and trigrams. Top phrases fill the remaining 50%.
3. **Greedy Encoding** — Longest-match-first via a Trie. Falls back to raw bytes for unknown sequences (zero OOV errors).

The C backend (`fast_counter` + `fast_encoder`) handles gigabyte-scale datasets.

## Quick Start

```bash
# Compile C backends
gcc -O3 -o fast_counter fast_counter.c
gcc -O3 -o fast_encoder fast_encoder.c

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
fast_counter.c         # C backend for counting atoms and n-grams
fast_encoder.c         # C backend for Trie-based greedy encoding
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
