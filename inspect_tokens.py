#!/usr/bin/env python3
"""Inspect tokenization of first 1000 chars under each Phase 9 config."""
import os
import subprocess
import time

from greedyphrase import GreedyPhraseTokenizer

DATA = "data/tiny_stories_100m"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_BIN = os.path.join(BASE_DIR, "fast_counter")
COUNTS = "tokenizer/counts.txt"

# Read first 1000 bytes
with open(DATA, 'rb') as f:
    snippet = f.read(1000)

# Write snippet for reference
with open("temp/snippet.txt", 'wb') as f:
    f.write(snippet)

configs = [
    ("baseline_bigram_2pass", dict(compound_passes=2, compound_max_n=2, template_budget=0)),
    ("raw_n7_converge",       dict(compound_max_n=7, compound_converge=True, template_budget=0)),
    ("hollow_n7",             dict(hollow=True, compound_max_n=7, template_budget=0)),
]

for name, kwargs in configs:
    print(f"\n{'='*60}")
    print(f"  Training: {name}")
    print(f"{'='*60}")

    vocab_path = f"temp/{name}.vocab"

    # Clean counts
    if os.path.exists(COUNTS):
        os.remove(COUNTS)

    # Count
    subprocess.run([COUNTER_BIN, DATA], check=True)

    # Train
    t = GreedyPhraseTokenizer(vocab_size=65536, model_path=vocab_path)
    t.train([DATA], **kwargs)
    t.save(vocab_path)
    t.load(vocab_path)  # rebuild trie

    # Encode snippet
    snippet_text = snippet.decode('utf-8', errors='replace')
    token_ids = t.encode(snippet_text)

    # Write token-by-token breakdown
    out_path = f"temp/{name}.tokens.txt"
    with open(out_path, 'w') as f:
        f.write(f"Config: {name}\n")
        f.write(f"Snippet: {len(snippet)} bytes, {len(token_ids)} tokens\n")
        f.write(f"Ratio: {len(snippet) / len(token_ids):.2f}x\n")
        f.write(f"\n--- Token breakdown (| = boundary) ---\n\n")

        # Show text with | delimiters between tokens
        parts = []
        for tid in token_ids:
            if 0 <= tid < len(t.vocab):
                parts.append(t.vocab[tid].decode('utf-8', errors='replace'))
            else:
                parts.append(f'<{tid}>')
        f.write('|'.join(parts))
        f.write('\n\n--- Token list ---\n\n')

        for i, tid in enumerate(token_ids):
            if 0 <= tid < len(t.vocab):
                tok_bytes = t.vocab[tid]
                tok_repr = repr(tok_bytes)[2:-1]  # strip b' and '
            else:
                tok_repr = f'<id={tid}>'
            f.write(f"  {i:4d}  id={tid:5d}  {tok_repr}\n")

    print(f"  Wrote {out_path}: {len(token_ids)} tokens ({len(snippet)/len(token_ids):.2f}x)")

print("\nDone. Inspect files in ./temp/")
