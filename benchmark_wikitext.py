#!/usr/bin/env python3
"""Benchmark GreedyPhrase vs tiktoken on WikiText-103-raw.

Clean Wikipedia prose — representative of LLM training data.
"""
import os
import subprocess
import time

from greedyphrase import GreedyPhraseTokenizer

DATA = "data/wikitext103.txt"
VOCAB = "tokenizer/greedyphrase.vocab"
COUNTS = "tokenizer/counts.txt"
OUTPUT = "data/wikitext103.tokens"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_BIN = os.path.join(BASE_DIR, "fast_counter")
ENCODER_BIN = os.path.join(BASE_DIR, "fast_encoder")

file_size = os.path.getsize(DATA)
print(f"WikiText-103-raw size: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")

# --- Compile ---
print("Compiling C backends...")
subprocess.run(
    ["gcc", "-O3", "-o", COUNTER_BIN, os.path.join(BASE_DIR, "fast_counter.c"), "-lpthread"],
    check=True,
)
subprocess.run(
    ["gcc", "-O3", "-o", ENCODER_BIN, os.path.join(BASE_DIR, "fast_encoder.c")],
    check=True,
)

results = []

# --- GreedyPhrase ---
print(f"\n{'='*60}")
print(f"  GreedyPhrase (65K vocab)")
print(f"{'='*60}")

for f in [VOCAB, COUNTS, OUTPUT]:
    if os.path.exists(f):
        os.remove(f)

print("  Counting...")
t0 = time.time()
subprocess.run([COUNTER_BIN, DATA], check=True)
count_time = time.time() - t0
print(f"  Count time: {count_time:.2f}s")

t = GreedyPhraseTokenizer(vocab_size=65536, model_path=VOCAB)
t.train([DATA])

print(f"\n  Encoding...")
t_enc = time.time()
t.encode_file(DATA, OUTPUT)
encode_time = time.time() - t_enc

token_file_size = os.path.getsize(OUTPUT)
num_tokens = token_file_size // 2
ratio = file_size / num_tokens

print(f"  Tokens:       {num_tokens:>12,}")
print(f"  Ratio:        {ratio:>12.2f}x")
print(f"  Encode time:  {encode_time:>12.2f}s")
results.append(("GreedyPhrase (65K)", num_tokens, ratio, encode_time))

# --- Tiktoken ---
import tiktoken

with open(DATA, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

for name, model in [("cl100k_base (GPT-4)", "cl100k_base"), ("o200k_base (GPT-4o)", "o200k_base")]:
    print(f"\n{'='*60}")
    print(f"  Tiktoken: {name}")
    print(f"{'='*60}")
    enc = tiktoken.get_encoding(model)
    start = time.time()
    tokens = enc.encode(text, disallowed_special=())
    elapsed = time.time() - start
    ratio = file_size / len(tokens)

    print(f"  Tokens:       {len(tokens):>12,}")
    print(f"  Ratio:        {ratio:>12.2f}x")
    print(f"  Encode time:  {elapsed:>12.2f}s")
    results.append((f"tiktoken {name}", len(tokens), ratio, elapsed))

# --- Summary ---
print(f"\n{'='*70}")
print(f"  WikiText-103-raw ({file_size/1e6:.0f} MB clean Wikipedia prose)")
print(f"{'='*70}")
print(f"  {'Tokenizer':<30} {'Tokens':>12} {'Ratio':>10} {'Enc MB/s':>10}")
print(f"  {'-'*62}")
for name, tokens, ratio, enc_time in results:
    mb_s = file_size / enc_time / 1e6
    print(f"  {name:<30} {tokens:>12,} {ratio:>10.2f}x {mb_s:>10.1f}")
print(f"{'='*70}")
