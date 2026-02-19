#!/usr/bin/env python3
"""Benchmark GreedyPhrase tokenizer on enwik9 without templates."""
import os
import time
from greedyphrase import GreedyPhraseTokenizer

DATA = "data/enwik9"
VOCAB = "tokenizer/greedyphrase_wiki_no_templates.vocab"
OUTPUT = "data/enwik9_no_templates.tokens"

if not os.path.exists(DATA):
    print(f"Data file {DATA} not found.")
    exit(1)

file_size = os.path.getsize(DATA)
print(f"enwik9 size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")

# Always retrain to pick up algorithm changes
if os.path.exists(VOCAB):
    os.remove(VOCAB)
    print(f"Deleted old vocab at {VOCAB}.")

t = GreedyPhraseTokenizer(vocab_size=65536, model_path=VOCAB)
t.train([DATA], template_budget=0)

# Encode with fast_encoder
print("\n--- Encoding enwik9 ---")
start = time.time()
t.encode_file(DATA, OUTPUT)
elapsed = time.time() - start

# Calculate compression ratio
token_file_size = os.path.getsize(OUTPUT)
num_tokens = token_file_size // 2  # uint16 = 2 bytes per token
compression_ratio = file_size / num_tokens

print(f"\n{'='*50}")
print(f"  GreedyPhrase Benchmark (enwik9) - No Templates")
print(f"{'='*50}")
print(f"  Input size:         {file_size:>15,} bytes")
print(f"  Vocab size:         {t.vocab_size:>15,}")
print(f"  Total tokens:       {num_tokens:>15,}")
print(f"  Compression ratio:  {compression_ratio:>15.2f}x (chars/token)")
print(f"  Encoding time:      {elapsed:>15.2f}s")
print(f"  Throughput:         {file_size / elapsed / 1e6:>15.2f} MB/s")
print(f"{'='*50}")
