#!/usr/bin/env python3
"""Count full sentence frequencies in TinyStories to evaluate macro token potential."""
import collections
import sys

DATA = "data/tiny_stories_100m"

print(f"Counting sentences in {DATA}...")
counts = collections.Counter()
n_sentences = 0

with open(DATA, 'rb') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split on sentence-ending punctuation while keeping the delimiter
        # Simple approach: split on ". " and "." at end, also "! " "? "
        text = line.decode('utf-8', errors='replace')
        # Walk through and extract sentences
        sent = []
        for char in text:
            sent.append(char)
            if char in '.!?' and len(sent) > 1:
                s = ''.join(sent).strip()
                if s:
                    counts[s] += 1
                    n_sentences += 1
                sent = []
        # Leftover (no terminal punctuation)
        s = ''.join(sent).strip()
        if s:
            counts[s] += 1
            n_sentences += 1

print(f"Total sentences: {n_sentences:,}")
print(f"Unique sentences: {len(counts):,}")

top = counts.most_common(5000)

# Stats
print(f"\nTop 5000 sentences:")
print(f"  Min freq: {top[-1][1]:,}")
print(f"  Max freq: {top[0][1]:,}")

total_matches = sum(freq for _, freq in top)
total_bytes_saved = sum(len(s.encode()) * freq for s, freq in top)
total_tokens_baseline = sum(len(s.encode()) // 9 * freq for s, freq in top)  # ~9.19 bytes/token
total_macro_tokens = total_matches  # 1 token per match

print(f"  Total matches: {total_matches:,}")
print(f"  Total bytes covered: {total_bytes_saved:,}")
print(f"  Baseline tokens (~9.19 b/t): {total_tokens_baseline:,}")
print(f"  Macro tokens (1 per match): {total_macro_tokens:,}")
print(f"  Tokens saved: {total_tokens_baseline - total_macro_tokens:,}")

print(f"\nTop 50 sentences:")
for i, (s, freq) in enumerate(top[:50]):
    print(f"  {i+1:3}. [{freq:>6,}x] ({len(s):>3} bytes) {s[:100]}")

# Length distribution of top 5000
lens = [len(s.encode()) for s, _ in top]
print(f"\nLength distribution of top 5000:")
print(f"  Min: {min(lens)} bytes")
print(f"  Max: {max(lens)} bytes")
print(f"  Avg: {sum(lens)/len(lens):.1f} bytes")
print(f"  Median: {sorted(lens)[len(lens)//2]} bytes")
