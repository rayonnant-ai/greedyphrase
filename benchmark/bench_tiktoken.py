#!/usr/bin/env python3
"""Benchmark tiktoken (GPT-4) on enwik9."""
import time
import os
import tiktoken

DATA = "data/enwik9"
file_size = os.path.getsize(DATA)

with open(DATA, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

for name, model in [("cl100k_base (GPT-4)", "cl100k_base"), ("o200k_base (GPT-4o)", "o200k_base")]:
    enc = tiktoken.get_encoding(model)
    start = time.time()
    tokens = enc.encode(text)
    elapsed = time.time() - start
    ratio = file_size / len(tokens)

    print(f"\n{'='*50}")
    print(f"  Tiktoken: {name}")
    print(f"{'='*50}")
    print(f"  Input size:         {file_size:>15,} bytes")
    print(f"  Vocab size:         {enc.n_vocab:>15,}")
    print(f"  Total tokens:       {len(tokens):>15,}")
    print(f"  Compression ratio:  {ratio:>15.2f}x (bytes/token)")
    print(f"  Encoding time:      {elapsed:>15.2f}s")
    print(f"  Throughput:         {file_size / elapsed / 1e6:>15.2f} MB/s")
    print(f"{'='*50}")
