#!/usr/bin/env python3
"""Benchmark GreedyPhrase tokenizer on TinyStories (100M subset).

Phase 8e: Two-stream canonical encoding — replace content words with \x01
placeholder, measure combined canonical-stream + fill-stream compression.
"""
import math
import os
import subprocess
import time

from greedyphrase import GreedyPhraseTokenizer

SRC = "/home/rrezel/src/korin/tiny_stories.txt"
DATA = "data/tiny_stories_100m"
CANON_DATA = "data/tiny_stories_100m.canon"
FILLS_DATA = "data/tiny_stories_100m.fills"
VOCAB = "tokenizer/greedyphrase.vocab"
COUNTS = "tokenizer/counts.txt"
OUTPUT = "data/tiny_stories_100m.tokens"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNTER_BIN = os.path.join(BASE_DIR, "fast_counter")
ENCODER_BIN = os.path.join(BASE_DIR, "fast_encoder")

# ─── Word sets from fast_mask.c ───

CONTENT_ADJECTIVES = {
    "big", "little", "small", "new", "old", "pretty", "happy", "sad",
    "scared", "brave", "kind", "nice", "mean", "silly", "funny", "curious",
    "gentle", "friendly", "soft", "hard", "bright", "dark", "warm", "cold",
    "sweet", "beautiful", "wonderful", "shiny", "special", "different",
    "favorite", "amazing", "important", "careful", "quiet", "loud", "tall",
    "short", "tiny", "huge", "fast", "slow", "real", "cool", "great",
    "best", "poor", "clean", "dirty", "wet", "dry", "hungry", "tired",
    "angry", "lonely", "excited", "proud",
    # adverbs
    "suddenly", "eventually", "finally", "quickly", "slowly", "carefully",
    "gently", "happily", "sadly", "quietly", "loudly", "bravely", "kindly",
    "softly", "eagerly", "politely", "proudly", "patiently", "cheerfully",
}

CONTENT_NOUNS = {
    "boy", "girl", "dog", "cat", "bird", "fish", "bear", "rabbit",
    "monkey", "elephant", "lion", "tiger", "fox", "frog", "mouse", "duck",
    "bunny", "puppy", "kitten", "pony", "ball", "toy", "cake", "apple",
    "flower", "tree", "house", "garden", "park", "forest", "river", "lake",
    "car", "boat", "hat", "dress", "shoe", "book", "box", "bag",
    "cup", "cookie", "candy", "water", "food", "gift", "star", "sun",
    "moon", "rain", "snow", "sand", "rock", "stick", "leaf", "door",
    "bed", "chair", "table", "princess", "prince", "king", "queen",
    "dragon", "fairy", "monster", "robot",
    # onomatopoeia
    "splash", "crash", "boom", "bang", "buzz", "hiss", "roar", "meow",
    "woof", "quack", "moo", "oink", "chirp", "squeak", "growl", "purr",
}

CONTENT_VERBS = {
    "play", "eat", "run", "build", "hide", "share", "catch", "throw",
    "pull", "push", "hold", "carry", "pick", "open", "close", "break",
    "fix", "sing", "dance", "draw", "paint", "swim", "jump", "climb",
    "fly", "sleep", "cry", "shout", "hug", "kiss", "wave", "kick",
    "bite", "dig", "cook",
    # dialogue verbs
    "said", "asked", "replied", "whispered", "shouted", "yelled",
    "called", "told", "answered", "exclaimed",
}

CONTENT_POSSESSIVES = {"his", "her", "their", "its"}

CONTENT_WORDS = CONTENT_ADJECTIVES | CONTENT_NOUNS | CONTENT_VERBS | CONTENT_POSSESSIVES

PROTECTED_WORDS = {
    "could", "should", "would", "can", "may", "might", "shall", "will", "must",
    "however", "therefore", "furthermore", "moreover", "meanwhile",
    "nevertheless", "although", "because", "since", "while", "though",
    "thus", "hence", "yet", "still", "also", "instead", "otherwise",
    "besides", "consequently", "accordingly", "subsequently",
    "between", "among", "through", "during", "before", "after",
    "above", "below", "across", "against", "along", "around",
    "beneath", "beside", "beyond", "despite", "except", "inside",
    "into", "near", "onto", "outside", "over", "past",
    "toward", "towards", "under", "underneath", "until", "upon",
    "within", "without", "about", "from", "with",
    "there", "here", "where", "when", "what", "which", "who",
    "how", "why", "that", "this", "these", "those",
    "some", "any", "each", "every", "both", "all", "such",
    "many", "much", "most", "more", "other", "another",
}

HONORIFICS = {
    "Dr.", "Prof.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.",
    "Lt.", "Col.", "Gen.", "Sgt.", "Cpl.", "Pvt.",
    "Rev.", "Hon.", "Capt.", "Cmdr.", "Adm.", "Maj.",
    "St.", "Gov.", "Pres.", "Sen.", "Rep.",
}

COMPASS_WORDS = {
    "North", "South", "East", "West",
    "Northeast", "Northwest", "Southeast", "Southwest",
    "North-east", "North-west", "South-east", "South-west",
    "NE", "NW", "SE", "SW",
    "NNE", "NNW", "SSE", "SSW", "ENE", "ESE", "WNW", "WSW",
}


# ─── Classification helpers (match fast_mask.c) ───

def _is_number(s):
    if not s or not s[0].isdigit():
        return False
    i = 1
    while i < len(s) and (s[i].isdigit() or s[i] == ','):
        i += 1
    if i < len(s) and s[i] == '.':
        i += 1
        while i < len(s) and s[i].isdigit():
            i += 1
    return i == len(s)


def _is_chemical(s):
    if len(s) < 2 or not s[0].isupper():
        return False
    has_digit = has_upper = False
    for c in s:
        if c.isupper():
            has_upper = True
        elif c.isdigit():
            has_digit = True
        elif c.islower() or c in ('(', ')'):
            pass
        else:
            return False
    return has_digit and has_upper


def _is_citation(s):
    if len(s) < 3 or s[0] != '[' or s[-1] != ']':
        return False
    has_digit = False
    for c in s[1:-1]:
        if c.isdigit():
            has_digit = True
        elif c not in (',', '-', ' '):
            return False
    return has_digit


def _is_list_marker(s):
    n = len(s)
    if n < 2 or n > 6:
        return False
    if s[0].isdigit() and s[-1] == '.':
        return all(c.isdigit() for c in s[:-1])
    if s[0].islower() and s[-1] in (')', '.') and n <= 4:
        return all(c.islower() for c in s[:-1])
    if s[0] == '(' and s[-1] == ')' and n >= 3:
        return all(c.islower() for c in s[1:-1])
    return False


def _should_mask(word, lc_set, cw_bytes, honor_bytes, compass_bytes):
    """Classify a word — returns True if it should be replaced with \\x01."""
    try:
        s = word.decode('ascii')
    except (UnicodeDecodeError, ValueError):
        return False

    # Match fast_mask.c classification order
    if _is_number(s):
        return True
    if _is_chemical(s):
        return True
    if _is_citation(s):
        return True
    if _is_list_marker(s):
        return True
    if word in honor_bytes:
        return True
    if word in compass_bytes:
        return True
    if '"' in s:
        return True

    # Content words + proper nouns (need lowercase comparison)
    lc = word.lower()
    if lc in cw_bytes:
        return True
    if len(word) > 1 and 65 <= word[0] <= 90:  # starts uppercase
        if lc not in lc_set:
            return True

    return False


def canonicalize(input_path, canon_path, fills_path):
    """Replace content words with \\x01, write canonical text + fills file.

    Two passes matching fast_mask.c logic:
      Pass 1: Build set of lowercase word forms (words starting lowercase)
      Pass 2: Classify each word; masked words become \\x01, originals go to fills
    """
    print(f"\n  Canonicalizing {input_path}...")
    t0 = time.time()

    with open(input_path, 'rb') as f:
        text = f.read()
    n = len(text)
    WS = frozenset({32, 9, 10, 13})

    # Pass 1: Build set of lowercase word forms (words starting lowercase)
    lc_set = set()
    pos = 0
    while pos < n:
        while pos < n and text[pos] in WS:
            pos += 1
        if pos >= n:
            break
        start = pos
        while pos < n and text[pos] not in WS:
            pos += 1
        if 97 <= text[start] <= 122:  # starts lowercase
            lc_set.add(text[start:pos].lower())

    # Add protected words to lc_set
    for w in PROTECTED_WORDS:
        lc_set.add(w.encode())

    print(f"  Pass 1: {len(lc_set):,} lowercase types ({time.time() - t0:.1f}s)")

    # Precompute byte sets for fast lookup
    cw_bytes = frozenset(w.encode() for w in CONTENT_WORDS)
    honor_bytes = frozenset(w.encode() for w in HONORIFICS)
    compass_bytes = frozenset(w.encode() for w in COMPASS_WORDS)

    # Pass 2: Classify and replace
    output = bytearray()
    fills = []
    fill_bytes = 0
    pos = 0

    while pos < n:
        # Copy whitespace
        ws_start = pos
        while pos < n and text[pos] in WS:
            pos += 1
        if pos > ws_start:
            output.extend(text[ws_start:pos])
        if pos >= n:
            break

        # Extract word
        start = pos
        while pos < n and text[pos] not in WS:
            pos += 1
        word = text[start:pos]

        if _should_mask(word, lc_set, cw_bytes, honor_bytes, compass_bytes):
            output.append(1)  # \x01
            fills.append(word)
            fill_bytes += len(word)
        else:
            output.extend(word)

    with open(canon_path, 'wb') as f:
        f.write(output)

    with open(fills_path, 'wb') as f:
        for fill in fills:
            f.write(fill)
            f.write(b'\n')

    elapsed = time.time() - t0
    print(f"  Pass 2: {len(fills):,} fills, {fill_bytes:,} fill bytes ({elapsed:.1f}s)")
    print(f"  Canon size: {len(output):,} bytes ({len(output) / n * 100:.1f}% of original)")

    return len(fills), fill_bytes


# ─── Data setup ───

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
    """Run a full train+encode pipeline and return (num_tokens, encode_time)."""
    print(f"\n{'=' * 60}")
    print(f"  Config: {name}")
    print(f"{'=' * 60}")

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

    print(f"\n  --- {name} Results ---")
    print(f"  Tokens:       {num_tokens:>12,}")
    print(f"  Encode time:  {encode_time:>12.2f}s")

    return num_tokens, encode_time


# ─── Run benchmarks ───

compile_binaries()

# Config 1: Baseline (2 passes) — original text
base_tokens, base_time = run_config(
    "Baseline (2 passes)",
    count_input=DATA,
    encode_input=DATA,
    template_budget=0,
    compound_passes=2,
)
base_ratio = file_size / base_tokens
print(f"  Ratio:        {base_ratio:>12.2f}x (original bytes/token)")

# Canonicalize the data
fill_count, fill_bytes_total = canonicalize(DATA, CANON_DATA, FILLS_DATA)
canon_size = os.path.getsize(CANON_DATA)

# Config 2: Canonical (2 passes)
c2_tokens, c2_time = run_config(
    "Canonical (2 passes)",
    count_input=CANON_DATA,
    encode_input=CANON_DATA,
    template_budget=0,
    compound_passes=2,
)
c2_canon_ratio = canon_size / c2_tokens
c2_eff_tokens = c2_tokens + math.ceil(fill_bytes_total / 2)
c2_eff_ratio = file_size / c2_eff_tokens
print(f"  Canon ratio:  {c2_canon_ratio:>12.2f}x (canon bytes/token)")
print(f"  Fills:        {fill_count:>12,} ({fill_bytes_total:,} bytes)")
print(f"  Eff tokens:   {c2_eff_tokens:>12,}")
print(f"  Eff ratio:    {c2_eff_ratio:>12.2f}x (original bytes/effective token)")

# Config 3: Canonical (4 passes)
c4_tokens, c4_time = run_config(
    "Canonical (4 passes)",
    count_input=CANON_DATA,
    encode_input=CANON_DATA,
    template_budget=0,
    compound_passes=4,
)
c4_canon_ratio = canon_size / c4_tokens
c4_eff_tokens = c4_tokens + math.ceil(fill_bytes_total / 2)
c4_eff_ratio = file_size / c4_eff_tokens
print(f"  Canon ratio:  {c4_canon_ratio:>12.2f}x (canon bytes/token)")
print(f"  Fills:        {fill_count:>12,} ({fill_bytes_total:,} bytes)")
print(f"  Eff tokens:   {c4_eff_tokens:>12,}")
print(f"  Eff ratio:    {c4_eff_ratio:>12.2f}x (original bytes/effective token)")

# ─── Summary ───

print(f"\n{'=' * 78}")
print(f"  Summary: TinyStories 100M — Phase 8e Two-Stream Canonical Encoding")
print(f"{'=' * 78}")
print(f"  {'Config':<25} {'Tokens':>10} {'Ratio':>8} {'Eff Tok':>10} {'Eff Ratio':>10} {'MB/s':>7}")
print(f"  {'-' * 72}")

mb_s = file_size / base_time / 1e6
print(f"  {'Baseline (2 passes)':<25} {base_tokens:>10,} {base_ratio:>7.2f}x {'—':>10} {'—':>10} {mb_s:>6.1f}")

mb_s = canon_size / c2_time / 1e6
print(f"  {'Canonical (2 passes)':<25} {c2_tokens:>10,} {c2_canon_ratio:>7.2f}x {c2_eff_tokens:>10,} {c2_eff_ratio:>9.2f}x {mb_s:>6.1f}")

mb_s = canon_size / c4_time / 1e6
print(f"  {'Canonical (4 passes)':<25} {c4_tokens:>10,} {c4_canon_ratio:>7.2f}x {c4_eff_tokens:>10,} {c4_eff_ratio:>9.2f}x {mb_s:>6.1f}")

print(f"  {'-' * 72}")
print(f"  Fills: {fill_count:,} words, {fill_bytes_total:,} bytes "
      f"(~{fill_bytes_total / fill_count:.1f} bytes/fill avg)")
print(f"  Effective token formula: canon_tokens + ceil(fill_bytes / 2)")
print(f"{'=' * 78}")
