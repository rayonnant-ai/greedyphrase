#!/usr/bin/env python3
"""Benchmark GreedyPhrase tokenizer on TinyStories (100M subset).

Plain English children's stories — no XML/wiki markup.
Tests how well phrase-based tokenization works on natural prose.
"""
import os
import subprocess
import time

from greedyphrase import GreedyPhraseTokenizer

SRC = "/home/rrezel/src/korin/tiny_stories.txt"
DATA = "data/tiny_stories_100m"
VOCAB = "tokenizer/greedyphrase.vocab"
COUNTS = "tokenizer/counts.txt"
OUTPUT = "data/tiny_stories_100m.tokens"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_BIN = os.path.join(BASE_DIR, "fast_counter")
ENCODER_BIN = os.path.join(BASE_DIR, "fast_encoder")

# Create 100M subset if needed
if not os.path.exists(DATA):
    print(f"Creating 100M subset from {SRC}...")
    with open(SRC, 'rb') as f:
        chunk = f.read(100_000_000)
    with open(DATA, 'wb') as f:
        f.write(chunk)
    print(f"Wrote {len(chunk):,} bytes to {DATA}")

file_size = os.path.getsize(DATA)
print(f"tiny_stories_100m size: {file_size:,} bytes ({file_size / 1e6:.1f} MB)")


def compile_binaries():
    print("Compiling C backends...")
    subprocess.run(
        ["gcc", "-O3", "-o", COUNTER_BIN, os.path.join(BASE_DIR, "fast_counter.c"), "-lpthread"],
        check=True,
    )
    subprocess.run(
        ["gcc", "-O3", "-o", ENCODER_BIN, os.path.join(BASE_DIR, "fast_encoder.c")],
        check=True,
    )


def run_config(name, count_input, encode_input, **train_kwargs):
    """Run a full train+encode pipeline and return (num_tokens, ratio, elapsed)."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    # Clean state
    for f in [VOCAB, COUNTS, OUTPUT]:
        if os.path.exists(f):
            os.remove(f)

    # Count
    print(f"  Counting on: {count_input}")
    t0 = time.time()
    subprocess.run([COUNTER_BIN, count_input], check=True)
    count_time = time.time() - t0
    print(f"  Count time: {count_time:.2f}s")

    # Train
    t = GreedyPhraseTokenizer(vocab_size=65536, model_path=VOCAB)
    t.train([count_input], **train_kwargs)

    # Encode
    print(f"\n  Encoding: {encode_input}")
    t_enc = time.time()
    t.encode_file(encode_input, OUTPUT)
    encode_time = time.time() - t_enc

    # Results
    token_file_size = os.path.getsize(OUTPUT)
    num_tokens = token_file_size // 2
    compression_ratio = file_size / num_tokens

    print(f"\n  --- {name} Results ---")
    print(f"  Original size:      {file_size:>15,} bytes")
    print(f"  Total tokens:       {num_tokens:>15,}")
    print(f"  Compression ratio:  {compression_ratio:>15.2f}x (original chars/token)")
    print(f"  Encoding time:      {encode_time:>15.2f}s")

    return name, num_tokens, compression_ratio, encode_time


compile_binaries()

results = []

r = run_config(
    "Phase8d no templates",
    count_input=DATA,
    encode_input=DATA,
    template_budget=0,
)
results.append(r)

r = run_config(
    "Phase8d + templates",
    count_input=DATA,
    encode_input=DATA,
    template_budget=2000,
)
results.append(r)

# Summary
print(f"\n{'='*70}")
print(f"  Summary: TinyStories 100M subset")
print(f"{'='*70}")
print(f"  {'Config':<30} {'Tokens':>12} {'Ratio':>10} {'Enc MB/s':>10}")
print(f"  {'-'*62}")
for name, tokens, ratio, enc_time in results:
    mb_s = file_size / enc_time / 1e6
    print(f"  {name:<30} {tokens:>12,} {ratio:>10.2f}x {mb_s:>10.1f}")
print(f"{'='*70}")
