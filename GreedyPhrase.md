# GreedyPhrase Tokenizer: Maximizing Effective Context via Greedy Phrase Compression

**Rohana Rezel**  
*Rayonnant AI*  
[rohana.rezel@rayonnant.ai](mailto:rohana.rezel@rayonnant.ai)

## Abstract

In the era of Large Language Models (LLMs), the efficiency of tokenization is a critical yet often overlooked factor in model performance. Standard Byte-Pair Encoding (BPE) tokenizers typically achieve compression ratios of 3-4x, limiting the effective context window and "intelligence density" of models, particularly those with smaller parameter counts. We introduce **GreedyPhrase**, a novel greedy dictionary-based tokenizer that dedicates 95% of its vocabulary to multi-word phrases and 5% to BPE subword tokens for residual coverage. By prioritizing frequent n-grams (bigrams and trigrams) and utilizing a deterministic longest-match-first strategy, GreedyPhrase achieves a compression ratio of **4.49x** on enwik9 (1 GB)—**1.23x better** than GPT-4's Tiktoken (3.65x) and **1.21x better** than GPT-4o's (3.70x)—while using a **1.5–3x smaller vocabulary** and maintaining 100% round-trip integrity via byte-level fallback. The optimized C backend encodes at 47 MB/s, completing a full train-and-encode cycle on 1GB in ~75 seconds.

## 1. Introduction

The tokenization bottleneck is a primary constraint in training efficient LLMs. While architectural innovations like Mixture of Experts (MoE) reduce computational cost, the tokenizer defines the fundamental unit of information the model processes. A suboptimal tokenizer forces the model to spend capacity learning low-level morphological compositions (e.g., "ing", "un", "likely") rather than high-level semantic concepts.

We propose that **"Compression is Intelligence"**: a tokenizer that compresses text into fewer, richer tokens allows the model to operate at a higher level of abstraction. The GreedyPhrase tokenizer achieves this by aggressively targeting multi-word phrases (e.g., "guest appearance", ", however,") as single atomic units.

## 2. Methodology

### 2.1 Dictionary Construction
Unlike BPE, which merges frequent character pairs iteratively, GreedyPhrase constructs its vocabulary in a three-stage process:
1.  **Atomic Tokenization & Phrase Mining:** The corpus is split into "atoms" (words, punctuation, whitespace) using a character-type classifier. We then mine the most frequent bigrams and trigrams. The top phrases fill 95% of available vocabulary slots.
2.  **First-Pass Encoding:** The corpus is encoded with the phrase vocabulary. Byte-fallback runs (sequences not covered by any phrase) are collected as residuals.
3.  **BPE on Residuals:** Standard BPE is trained on the residual byte sequences. The resulting subword tokens fill the remaining 5% of vocabulary, capturing morphological patterns the phrase layer misses.

This hybrid approach ensures coverage of high-frequency semantic phrases while using BPE to efficiently handle rare words, markup, and morphological variation.

### 2.2 Greedy Encoding with Trie Optimization
Encoding is performed using a **Longest-Match-First** strategy. A Trie (prefix tree) is constructed from the vocabulary. For any given position in the text, the tokenizer greedily selects the longest possible substring that matches a valid token.
- **Deterministic:** The output is always consistent for a given text.
- **Robust:** If no match is found (rare), the tokenizer falls back to byte-level tokens (reserved IDs 0-255), ensuring zero Out-Of-Vocabulary (OOV) errors.

### 2.3 C-Optimized Implementation
Naive Python implementations of this greedy strategy are prohibitively slow (~10k tokens/sec). We developed a custom C backend with several layers of optimization:
- **Fast Counter:** Multithreaded (12 cores) n-gram counting with xxHash (XXH3_64bits), per-thread chained hash tables (4M buckets each), arena allocation, and post-hoc merge into 128M-bucket global tables. Processes enwik9 at **81 MB/s** (counting phase).
- **Fast Encoder:** `mmap`-based input with `MADV_SEQUENTIAL`, contiguous trie node pool (int32 indices instead of scattered pointers), 1MB buffered output, and speculative `_mm_prefetch` during trie traversal. Encodes at **47 MB/s**.

## 3. Experiments and Results

We benchmarked GreedyPhrase against industry-standard tokenizers on **enwik9** (1.00 GB, raw Wikipedia XML dump)—a standard compression benchmark containing natural language, XML markup, URLs, and other non-natural-language content.

### 3.1 Quantitative Analysis

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio (Bytes/Token) |
| :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **222,805,405** | **4.49x** |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 270,616,861 | 3.70x |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 273,662,103 | 3.65x |

GreedyPhrase outperforms GPT-4's tokenizer by **1.23x** and GPT-4o's by **1.21x** on enwik9, while using a vocabulary **1.5–3x smaller**. The hybrid phrase+BPE architecture handles enwik9's heavy XML markup effectively: phrases capture repeated structural patterns while BPE handles rare tokens and morphological variation.

For a fixed context window of 2048 tokens, GreedyPhrase encodes ~9,200 bytes of enwik9, whereas GPT-4's Tiktoken encodes only ~7,500—a **23% effective context advantage**.

### 3.2 Phrase Ratio Impact
We analyzed the impact of the "Phrase Ratio" (percentage of vocab dedicated to multi-word phrases) on compression efficiency:

| Phrase Ratio | Compression Ratio |
| :--- | :--- |
| 10% | 4.94x |
| 30% | 5.68x |
| 50% | 6.04x |
| **70%** | **6.28x (Optimal)** |
| 90% | 6.21x |

The sweet spot was found at **70%**, demonstrating the immense value of multi-word tokens in natural language.

## 4. Discussion and Future Work

The GreedyPhrase tokenizer represents a shift from *subword* modeling to *superword* (phrase) modeling. By encoding "guest appearance" as a single integer ID, the model's embedding layer learns a unique vector for that specific concept, rather than composing it from "guest" + " appear" + "ance". This allows smaller models (e.g., 1B parameters) to exhibit reasoning capabilities typically associated with larger models, as they are effectively "standing on the shoulders" of a smarter tokenizer.

## 5. Conclusion

We presented GreedyPhrase, a high-performance tokenizer designed for efficient LLM training. Its greedy phrase-based approach yields state-of-the-art compression ratios, and its optimized C implementation ensures it scales to massive datasets. GreedyPhrase serves as the foundation for the GreedyPhrase-1B model, proving that smart data processing is as vital as model architecture.

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
