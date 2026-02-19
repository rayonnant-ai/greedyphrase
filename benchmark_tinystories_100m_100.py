#!/usr/bin/env python3
"""Benchmark GreedyPhrase tokenizer on tiny_stories_100m with template_budget=100."""
import os
import time
from greedyphrase import GreedyPhraseTokenizer

DATA = "data/tiny_stories_100m"
VOCAB = "tokenizer/greedyphrase_ts_100.vocab"
OUTPUT = "data/tiny_stories_100m_100.tokens"

if not os.path.exists(DATA):
    print(f"Data file {DATA} not found.")
    exit(1)

file_size = os.path.getsize(DATA)
print(f"tiny_stories_100m size: {file_size:,} bytes ({file_size / 1e6:.2f} MB)")

# Always retrain to pick up algorithm changes
if os.path.exists(VOCAB):
    os.remove(VOCAB)
    print(f"Deleted old vocab at {VOCAB}.")

t = GreedyPhraseTokenizer(vocab_size=65536, model_path=VOCAB)
# Set template_budget=100
t.train([DATA], template_budget=100)

# Encode with fast_encoder
print("\n--- Encoding tiny_stories_100m ---")
start = time.time()
t.encode_file(DATA, OUTPUT)
elapsed = time.time() - start

# Calculate compression ratio
token_file_size = os.path.getsize(OUTPUT)
num_tokens = token_file_size // 2  # uint16 = 2 bytes per token
compression_ratio = file_size / num_tokens

print(f"\n{'='*50}")
print(f"  GreedyPhrase Benchmark (tiny_stories_100m) - Budget 100")
print(f"{'='*50}")
print(f"  Input size:         {file_size:>15,} bytes")
print(f"  Vocab size:         {t.vocab_size:>15,}")
print(f"  Total tokens:       {num_tokens:>15,}")
print(f"  Compression ratio:  {compression_ratio:>15.2f}x (chars/token)")
print(f"  Encoding time:      {elapsed:>15.2f}s")
print(f"  Throughput:         {file_size / elapsed / 1e6:>15.2f} MB/s")
print(f"{'='*50}")
