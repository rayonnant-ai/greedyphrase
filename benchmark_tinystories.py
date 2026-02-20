#!/usr/bin/env python3
"""Benchmark GreedyPhrase tokenizer on TinyStories (100M subset).

Plain English children's stories — no XML/wiki markup.
Tests how well phrase-based tokenization works on natural prose.

Phase 11: Compares baseline, suffix-array mining, and event model tokenization.
"""
import os
import subprocess
import time

from greedyphrase import GreedyPhraseTokenizer
from event_parser import EventParser

SRC = "/home/rrezel/src/korin/tiny_stories.txt"
DATA = "data/tiny_stories_100m"
VOCAB = "tokenizer/greedyphrase.vocab"
COUNTS = "tokenizer/counts.txt"
OUTPUT = "data/tiny_stories_100m.tokens"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_BIN = os.path.join(BASE_DIR, "fast_counter")
ENCODER_BIN = os.path.join(BASE_DIR, "fast_encoder")
MINER_BIN = os.path.join(BASE_DIR, "phase10_miner")


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
    subprocess.run(
        ["gcc", "-O3", "-mavx2", "-o", MINER_BIN, os.path.join(BASE_DIR, "phase10_miner.c")],
        check=True,
    )


def run_config(name, **train_kwargs):
    """Run a full train+encode pipeline on DATA and return result dict."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    # Clean state
    for f in [VOCAB, COUNTS, OUTPUT]:
        if os.path.exists(f):
            os.remove(f)

    # Count
    print(f"  Counting on: {DATA}")
    t0 = time.time()
    subprocess.run([COUNTER_BIN, DATA], check=True)
    count_time = time.time() - t0
    print(f"  Count time: {count_time:.2f}s")

    # Train
    t = GreedyPhraseTokenizer(vocab_size=65536, model_path=VOCAB)
    t.train([DATA], **train_kwargs)

    # Encode
    print(f"\n  Encoding: {DATA}")
    t_enc = time.time()
    t.encode_file(DATA, OUTPUT)
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

    return {
        'name': name,
        'tokens': num_tokens,
        'ratio': compression_ratio,
        'enc_time': encode_time,
    }


def run_phase10(name, min_freq=50, max_len=200, vocab_size=65536):
    """Run Phase 10 suffix-array miner + fast_encoder pipeline."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    # Clean state
    for f in [VOCAB, OUTPUT]:
        if os.path.exists(f):
            os.remove(f)

    # Mine vocab
    print(f"  Mining vocab from: {DATA}")
    t0 = time.time()
    subprocess.run([
        MINER_BIN, DATA,
        "-o", VOCAB,
        "-v", str(vocab_size),
        "-f", str(min_freq),
        "-l", str(max_len),
    ], check=True)
    mine_time = time.time() - t0
    print(f"  Mine time: {mine_time:.2f}s")

    # Encode with fast_encoder
    print(f"\n  Encoding: {DATA}")
    t_enc = time.time()
    subprocess.run([ENCODER_BIN, VOCAB, DATA, OUTPUT], check=True)
    encode_time = time.time() - t_enc

    # Results
    token_file_size = os.path.getsize(OUTPUT)
    num_tokens = token_file_size // 2
    compression_ratio = file_size / num_tokens

    print(f"\n  --- {name} Results ---")
    print(f"  Original size:      {file_size:>15,} bytes")
    print(f"  Total tokens:       {num_tokens:>15,}")
    print(f"  Compression ratio:  {compression_ratio:>15.2f}x (original chars/token)")
    print(f"  Mine time:          {mine_time:>15.2f}s")
    print(f"  Encoding time:      {encode_time:>15.2f}s")

    return {
        'name': name,
        'tokens': num_tokens,
        'ratio': compression_ratio,
        'enc_time': encode_time,
        'mine_time': mine_time,
    }


EVENT_DATA = "data/tiny_stories_100m_event.txt"
EVENT_VOCAB = "tokenizer/greedyphrase_event.vocab"
EVENT_OUTPUT = "data/tiny_stories_100m_event.tokens"


def run_event_model(name, **train_kwargs):
    """Run event model pipeline: preprocess -> train -> encode."""
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    # Step 1: Preprocess to event model text (cached)
    if not os.path.exists(EVENT_DATA):
        print(f"  Transforming TinyStories to event model text...")
        t0 = time.time()
        text = open(DATA, 'r', encoding='utf-8', errors='replace').read()
        parser = EventParser()
        event_text = parser.transform(text)
        with open(EVENT_DATA, 'w', encoding='utf-8') as f:
            f.write(event_text)
        parse_time = time.time() - t0
        print(f"  Parse time: {parse_time:.2f}s")
    else:
        print(f"  Using cached event text: {EVENT_DATA}")

    event_size = os.path.getsize(EVENT_DATA)
    print(f"  Event text size: {event_size:,} bytes ({event_size/file_size*100:.1f}% of original)")

    # Clean state
    for f in [EVENT_VOCAB, COUNTS, EVENT_OUTPUT]:
        if os.path.exists(f):
            os.remove(f)

    # Count on event text
    print(f"  Counting on: {EVENT_DATA}")
    t0 = time.time()
    subprocess.run([COUNTER_BIN, EVENT_DATA], check=True)
    count_time = time.time() - t0
    print(f"  Count time: {count_time:.2f}s")

    # Train on event text
    t = GreedyPhraseTokenizer(vocab_size=65536, model_path=EVENT_VOCAB)
    t.train([EVENT_DATA], **train_kwargs)

    # Encode event text
    print(f"\n  Encoding: {EVENT_DATA}")
    t_enc = time.time()
    t.encode_file(EVENT_DATA, EVENT_OUTPUT)
    encode_time = time.time() - t_enc

    # Results
    token_file_size = os.path.getsize(EVENT_OUTPUT)
    num_tokens = token_file_size // 2
    event_ratio = event_size / num_tokens
    effective_ratio = file_size / num_tokens

    print(f"\n  --- {name} Results ---")
    print(f"  Original size:      {file_size:>15,} bytes")
    print(f"  Event text size:    {event_size:>15,} bytes ({event_size/file_size*100:.1f}%)")
    print(f"  Total tokens:       {num_tokens:>15,}")
    print(f"  Event compression:  {event_ratio:>15.2f}x (event bytes/token)")
    print(f"  Effective ratio:    {effective_ratio:>15.2f}x (original bytes/token)")
    print(f"  Encoding time:      {encode_time:>15.2f}s")

    return {
        'name': name,
        'tokens': num_tokens,
        'ratio': effective_ratio,
        'event_ratio': event_ratio,
        'event_size': event_size,
        'enc_time': encode_time,
    }


compile_binaries()

results = []

# Config 1: Baseline (bigram, 2-pass)
r = run_config(
    "Baseline (bigram, 2-pass)",
    compound_passes=2, compound_max_n=2,
    template_budget=0,
)
results.append(r)

# Config 2: Phase 10 — Suffix Array Greedy Mining
r = run_phase10("Phase10 SA-Greedy (f=50)")
results.append(r)

# Config 3: Phase 10 — lower min_freq
r = run_phase10("Phase10 SA-Greedy (f=20)", min_freq=20)
results.append(r)

# Config 4: Event Model
r = run_event_model(
    "Event Model (bigram, 2-pass)",
    compound_passes=2, compound_max_n=2,
    template_budget=0,
)
results.append(r)

# Summary
print(f"\n{'='*70}")
print(f"  Summary: TinyStories 100M subset ({file_size:,} bytes)")
print(f"{'='*70}")
print(f"  {'Config':<30} {'Tokens':>12} {'Ratio':>10} {'Enc MB/s':>10}")
print(f"  {'-'*62}")
for r in results:
    mb_s = file_size / r['enc_time'] / 1e6
    extra = ""
    if 'event_ratio' in r:
        extra = f"  (event: {r['event_ratio']:.2f}x, {r['event_size']/file_size*100:.0f}% size)"
    print(f"  {r['name']:<30} {r['tokens']:>12,} {r['ratio']:>10.2f}x {mb_s:>10.1f}{extra}")
print(f"{'='*70}")
