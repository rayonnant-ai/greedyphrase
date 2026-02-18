# GreedyPhrase Tokenizer: Maximizing Effective Context via Greedy Phrase Compression

**Rohana Rezel**  
*Rayonnant AI*  
[rohana.rezel@rayonnant.ai](mailto:rohana.rezel@rayonnant.ai)

## Abstract

In the era of Large Language Models (LLMs), the efficiency of tokenization is a critical yet often overlooked factor in model performance. Standard Byte-Pair Encoding (BPE) tokenizers typically achieve compression ratios of 3-4x, limiting the effective context window and "intelligence density" of models, particularly those with smaller parameter counts. We introduce **GreedyPhrase**, a novel greedy dictionary-based tokenizer that uses iterative compound training to build a vocabulary of multi-word phrases (up to 21+ atoms long), augmented by BPE for residual coverage. GreedyPhrase achieves a compression ratio of **6.04x** on WikiText-103 (539 MB)—**34% better** than GPT-4's Tiktoken (4.49x) and GPT-4o's (4.53x)—and **9.18x** on TinyStories—**2.24x better** than Tiktoken (4.07x)—while using a **1.5–3x smaller vocabulary** and maintaining 100% round-trip integrity via byte-level fallback. The optimized C backend encodes at 37-43 MB/s, 3-6x faster than Tiktoken.

## 1. Introduction

The tokenization bottleneck is a primary constraint in training efficient LLMs. While architectural innovations like Mixture of Experts (MoE) reduce computational cost, the tokenizer defines the fundamental unit of information the model processes. A suboptimal tokenizer forces the model to spend capacity learning low-level morphological compositions (e.g., "ing", "un", "likely") rather than high-level semantic concepts.

We propose that **"Compression is Intelligence"**: a tokenizer that compresses text into fewer, richer tokens allows the model to operate at a higher level of abstraction. The GreedyPhrase tokenizer achieves this by aggressively targeting multi-word phrases (e.g., "guest appearance", ", however,") as single atomic units.

## 2. Methodology

### 2.1 Dictionary Construction
Unlike BPE, which merges frequent character pairs iteratively, GreedyPhrase constructs its vocabulary using an iterative compound training process:
1.  **Atomic Tokenization & Primitive Phrase Mining:** The corpus is split into "atoms" (words, punctuation, whitespace) using a character-type classifier. We count all n-grams up to 7 atoms long. The top ~52K phrases by frequency become the primitive vocabulary.
2.  **Compound Pass 1:** The corpus is encoded with the primitive vocabulary. We then count consecutive token-ID pairs (bigrams) in the encoded stream. Each bigram of two primitive tokens produces a compound phrase by concatenating their byte sequences — e.g., a bigram of two 5-atom tokens yields a 10-atom compound. Compounds are scored by `frequency × byte_length` and the top ~5K are selected.
3.  **Compound Pass 2:** The corpus is re-encoded with the expanded vocabulary (primitives + pass-1 compounds). Token bigrams are counted again. A bigram of two compound tokens can yield a "triple compound" — e.g., two 7-atom compounds yield a 14-atom phrase, or a 7-atom primitive paired with a 14-atom compound yields a 21-atom phrase. The top ~5K are selected.
4.  **Residual Collection:** The corpus is encoded once more with the full vocabulary (primitives + all compounds). Byte-fallback runs are collected as residuals.
5.  **BPE on Residuals:** Standard BPE is trained on the residual byte sequences. The resulting ~3K subword tokens fill the remaining vocabulary slots, capturing morphological patterns the phrase layer misses.

The key insight is that each compounding pass doubles the maximum phrase reach. A 21-atom phrase is simply two tokens (one 7-atom primitive and one 14-atom compound) appearing consecutively. By iterating encode-then-count-bigrams, we capture arbitrarily long phrases without the combinatorial explosion of high-order n-gram counting (which would OOM on large corpora). Experimentally, two compounding passes (3 total encoding passes) hit the sweet spot — a fourth pass adds less than 1% compression.

### 2.2 Greedy Encoding with Trie Optimization
Encoding is performed using a **Longest-Match-First** strategy. A Trie (prefix tree) is constructed from the vocabulary. For any given position in the text, the tokenizer greedily selects the longest possible substring that matches a valid token.
- **Deterministic:** The output is always consistent for a given text.
- **Robust:** If no match is found (rare), the tokenizer falls back to byte-level tokens (reserved IDs 0-255), ensuring zero Out-Of-Vocabulary (OOV) errors.

### 2.3 C-Optimized Implementation
Naive Python implementations of this greedy strategy are prohibitively slow (~10k tokens/sec). We developed a custom C backend with several layers of optimization:
- **Fast Counter:** Multithreaded (12 cores) n-gram counting (up to 7-grams) with xxHash (XXH3_64bits), per-thread chained hash tables (8M buckets each), arena allocation, and post-hoc merge into 128M-bucket global tables.
- **Fast Encoder:** `mmap`-based input with `MADV_SEQUENTIAL`, contiguous trie node pool (int32 indices instead of scattered pointers), 1MB buffered output, and speculative `_mm_prefetch` during trie traversal. Encodes at **33-45 MB/s** depending on vocabulary size.
- **Token Bigram Counter:** Numpy-based vectorized counting of consecutive uint16 token pairs. Encodes pairs as `(a << 16) | b` into uint32 for fast `np.unique`-based counting. Processes 100M+ tokens in seconds.

## 3. Experiments and Results

We benchmarked GreedyPhrase against industry-standard Tiktoken tokenizers on two datasets representing clean natural language: **WikiText-103-raw** (539 MB, clean Wikipedia prose) and **TinyStories** (100 MB, simple English children's stories).

### 3.1 WikiText-103-raw (539 MB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Enc MB/s |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **89,291,627** | **6.04x** | **42.5** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 120,196,189 | 4.49x | 11.9 |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 119,160,774 | 4.53x | 7.1 |

GreedyPhrase achieves **34% better compression** than both GPT-4 and GPT-4o tokenizers, while using a vocabulary **1.5–3x smaller** and encoding **3-6x faster**.

For a fixed context window of 2048 tokens, GreedyPhrase encodes ~12,400 bytes of WikiText, whereas Tiktoken encodes only ~9,200—a **34% effective context advantage**.

### 3.2 TinyStories (100 MB)

| Tokenizer | Vocab Size | Total Tokens | Compression Ratio | Enc MB/s |
| :--- | :--- | :--- | :--- | :--- |
| **GreedyPhrase (Ours)** | **65,536** | **10,890,713** | **9.18x** | **36.9** |
| Tiktoken cl100k_base (GPT-4) | 100,277 | 24,541,816 | 4.07x | 10.9 |
| Tiktoken o200k_base (GPT-4o) | 200,019 | 24,367,822 | 4.10x | 6.9 |

On repetitive natural prose, GreedyPhrase achieves **2.24x better compression** than Tiktoken. The phrase-based approach excels when the same multi-word patterns recur frequently — exactly the scenario encountered in training data for small language models.

### 3.3 Vocabulary Budget Allocation

The iterative compound training splits the 65K vocabulary into four tiers:

| Tier | Slots | Role |
| :--- | :--- | :--- |
| Primitives | ~52K | Frequent n-grams up to 7 atoms (direct counting) |
| Compounds (pass 1) | ~5K | Token bigrams up to 14 atoms |
| Compounds (pass 2) | ~5K | Token bigrams up to 21+ atoms |
| BPE | ~3K | Subword tokens for residual byte sequences |

The compound tiers are critical: they capture long phrases like "once upon a time, there was a little" that far exceed the 7-atom counting window, without the memory cost of high-order n-gram hash tables. Experimentally, two compounding passes hit diminishing returns — a third adds less than 1%.

## 4. Discussion

The GreedyPhrase tokenizer represents a shift from *subword* modeling to *superword* (phrase) modeling. By encoding "guest appearance" as a single integer ID, the model's embedding layer learns a unique vector for that specific concept, rather than composing it from "guest" + " appear" + "ance". This allows smaller models (e.g., 1B parameters) to exhibit reasoning capabilities typically associated with larger models, as they are effectively "standing on the shoulders" of a smarter tokenizer.

The iterative compound approach is key to scaling phrase length without scaling memory. Directly counting 21-grams on 539 MB of WikiText would require hundreds of GB of hash arena, causing OOM on commodity hardware. The iterative approach achieves the same effective n-gram reach with cheap passes: one counting 7-grams (manageable), then two rounds of counting bigrams of token IDs (trivial — numpy vectorized in seconds each).

The compression advantage is most pronounced on repetitive natural prose (9.18x on TinyStories vs 4.07x for Tiktoken). This suggests GreedyPhrase is particularly well-suited for domain-specific models trained on corpora with recurring phrases — legal documents, medical records, code, and structured text.

## 5. Conclusion

We presented GreedyPhrase, a high-performance tokenizer designed for efficient LLM training. Its iterative compound phrase approach yields state-of-the-art compression ratios (6.04x on WikiText-103, 9.18x on TinyStories) while using a vocabulary 1.5-3x smaller than standard BPE tokenizers. The optimized C implementation ensures it scales to gigabyte datasets. GreedyPhrase serves as the foundation for the GreedyPhrase-1B model, proving that smart data processing is as vital as model architecture.

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
