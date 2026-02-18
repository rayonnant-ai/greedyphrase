import os
import struct
import collections
import tempfile
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np


SLOT_SENTINEL = 0xFFFF
TEMPLATE_HASH_PRIME = np.uint64(1000000007)


@dataclass
class TemplateInfo:
    """Metadata for a single 1-slot word-level template pattern."""
    vocab_id: int              # assigned ID in the vocab (>= len(base_vocab))
    num_words: int             # number of words in the template (including slot)
    slot_pos: int              # which word is the slot (0-indexed)
    word_strings: List[Optional[str]] = field(default_factory=list)  # word strings (None at slot_pos)
    bytes_before_fill: bytes = b""   # for decoding: literal bytes before the fill token
    bytes_after_fill: bytes = b""    # for decoding: literal bytes after the fill token


def train_bpe(segments: List[bytes], num_merges: int) -> List[bytes]:
    """
    Train BPE on a list of byte segments and return merged tokens.
    Each segment is a bytes object; segment boundaries are never crossed.
    """
    # Convert segments to lists of single-byte tokens, skip length-1
    seqs = []
    for seg in segments:
        if len(seg) > 1:
            seqs.append(list(seg))  # list of ints (0-255)

    if not seqs:
        return []

    # Build initial pair counts and index of which seqs contain each pair
    pair_counts = collections.Counter()
    # pair_index[pair] = set of seq indices that contain this pair
    pair_index = collections.defaultdict(set)

    for si, seq in enumerate(seqs):
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += 1
            pair_index[pair].add(si)

    merged_tokens = []

    for _ in range(num_merges):
        if not pair_counts:
            break
        # Find most frequent pair
        best_pair = pair_counts.most_common(1)[0]
        pair, count = best_pair
        if count < 2:
            break

        # The new merged token as bytes
        if isinstance(pair[0], int):
            left = bytes([pair[0]])
        else:
            left = pair[0]
        if isinstance(pair[1], int):
            right = bytes([pair[1]])
        else:
            right = pair[1]
        new_token = left + right
        merged_tokens.append(new_token)

        # Merge this pair in all sequences that contain it
        affected = pair_index.pop(pair, set())
        del pair_counts[pair]

        for si in affected:
            seq = seqs[si]
            i = 0
            while i < len(seq) - 1:
                if seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    # Remove old pairs involving positions i-1,i and i+1,i+2
                    if i > 0:
                        old_left = (seq[i - 1], seq[i])
                        pair_counts[old_left] -= 1
                        if pair_counts[old_left] <= 0:
                            del pair_counts[old_left]
                        pair_index[old_left].discard(si)

                    if i + 2 < len(seq):
                        old_right = (seq[i + 1], seq[i + 2])
                        pair_counts[old_right] -= 1
                        if pair_counts[old_right] <= 0:
                            del pair_counts[old_right]
                        pair_index[old_right].discard(si)

                    # Replace pair with new_token
                    seq[i] = new_token
                    del seq[i + 1]

                    # Add new pairs
                    if i > 0:
                        new_left = (seq[i - 1], seq[i])
                        pair_counts[new_left] += 1
                        pair_index[new_left].add(si)
                    if i + 1 < len(seq):
                        new_right = (seq[i], seq[i + 1])
                        pair_counts[new_right] += 1
                        pair_index[new_right].add(si)
                    # Don't advance i — check for consecutive merges
                else:
                    i += 1

    # Convert merged tokens: they may be nested (bytes objects from merges)
    # Flatten each to a plain bytes
    result = []
    for t in merged_tokens:
        if isinstance(t, bytes):
            result.append(t)
        else:
            # Should not happen, but just in case
            result.append(bytes([t]) if isinstance(t, int) else t)
    return result

def count_token_bigrams(tokens_path, min_freq=100):
    """Count consecutive token ID pairs from a .tokens file using numpy.

    Encodes pairs as (a << 16) | b into uint32 for fast np.unique counting.
    Returns dict of (id_a, id_b) -> count, filtered by min_freq.
    """
    tokens = np.fromfile(tokens_path, dtype=np.uint16)
    if len(tokens) < 2:
        return {}
    pairs = (tokens[:-1].astype(np.uint32) << 16) | tokens[1:].astype(np.uint32)
    unique_pairs, counts = np.unique(pairs, return_counts=True)
    mask = counts >= min_freq
    result = {}
    for packed, count in zip(unique_pairs[mask], counts[mask]):
        id_a = int(packed >> 16)
        id_b = int(packed & 0xFFFF)
        result[(id_a, id_b)] = int(count)
    return result


def mine_templates(tokens_path: str, template_budget: int = 2000,
                    min_freq: int = 100, min_L: int = 4) -> List[Tuple[List[int], int, float]]:
    """Mine 1-slot templates from a token stream using polynomial hashing.

    Returns list of (frame, slot_position, score) sorted by score descending,
    where frame is a list of token IDs with SLOT_SENTINEL at the slot position.
    """
    tokens = np.fromfile(tokens_path, dtype=np.uint16)
    n = len(tokens)
    print(f"  Mining templates from {n:,} tokens (L={min_L}..10, min_freq={min_freq})...",
          flush=True)

    # Precompute powers of PRIME
    max_len = 11
    powers = np.array([TEMPLATE_HASH_PRIME ** np.uint64(j) for j in range(max_len)],
                      dtype=np.uint64)

    candidates = []  # (score, frame_tuple, slot_pos)

    for L in range(min_L, 11):
        if n < L:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(tokens, L)  # (N-L+1, L)
        num_windows = windows.shape[0]

        # Full polynomial hash: Σ(windows[:, j] * PRIME^j)
        h_full = np.zeros(num_windows, dtype=np.uint64)
        for j in range(L):
            h_full += windows[:, j].astype(np.uint64) * powers[j]

        for k in range(1, L):  # slot position (skip 0)
            # Adjust hash: replace token at position k with SLOT_SENTINEL
            h_slot = h_full + (np.uint64(SLOT_SENTINEL) - windows[:, k].astype(np.uint64)) * powers[k]

            unique_hashes, first_idx, counts = np.unique(
                h_slot, return_index=True, return_counts=True)

            mask = counts >= min_freq
            for ui in np.where(mask)[0]:
                freq = int(counts[ui])
                # saved = L - 1 (frame becomes 1 template_id + 1 fill) - but 1 slot
                # net tokens saved per match = L - 2
                score = (L - 2) * freq
                wi = int(first_idx[ui])
                frame = list(windows[wi].astype(int))
                frame[k] = SLOT_SENTINEL
                candidates.append((score, tuple(frame), k))

        print(f"    L={L}: {len(candidates):,} cumulative candidates", flush=True)

    # Sort by score descending, dedup by frame
    candidates.sort(key=lambda x: -x[0])
    seen_frames = set()
    result = []
    for score, frame_tuple, slot_pos in candidates:
        if frame_tuple in seen_frames:
            continue
        seen_frames.add(frame_tuple)
        result.append((list(frame_tuple), slot_pos, score))
        if len(result) >= template_budget:
            break

    print(f"  Selected {len(result)} templates.", flush=True)
    return result


class GreedyPhraseTokenizer:
    """
    A custom greedy dictionary-based tokenizer.
    
    Features:
    - Fixed dictionary size (e.g., 65536 tokens).
    - Top 50% of dictionary reserved for frequent multi-word phrases.
    - Remaining 50% for standard words/subwords.
    - Deterministic, greedy matching (longest match first).
    - Outputs uint16 tokens.
    """
    def __init__(self, vocab_size=65536, model_path="tokenizer/greedyphrase.vocab"):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.vocab: List[bytes] = []
        self.token_to_id: Dict[bytes, int] = {}
        self.trie = {} # Character-based trie for fast greedy matching
        self.templates: Dict[int, TemplateInfo] = {}  # vocab_id -> TemplateInfo

        if os.path.exists(model_path):
            self.load(model_path)
            
    @staticmethod
    def _save_vocab(vocab: List[bytes], path: str):
        """Save a vocab list to the length-prefixed binary format."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(struct.pack('>I', len(vocab)))
            for token in vocab:
                f.write(struct.pack('>I', len(token)))
                f.write(token)

    def _ensure_fast_encoder(self):
        """Compile fast_encoder if needed and return its path."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        encoder_bin = os.path.join(base_dir, "fast_encoder")
        if not os.path.exists(encoder_bin):
            print("Compiling fast_encoder...", flush=True)
            subprocess.run(
                ["gcc", "-O3", "-o", encoder_bin, os.path.join(base_dir, "fast_encoder.c")],
                check=True
            )
        return encoder_bin

    @staticmethod
    def _ensure_c_binary(name, extra_flags=None):
        """Compile a C binary if needed and return its path."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        bin_path = os.path.join(base_dir, name)
        src_path = bin_path + '.c'
        if not os.path.exists(bin_path) or os.path.getmtime(src_path) > os.path.getmtime(bin_path):
            print(f"Compiling {name}...", flush=True)
            cmd = ['gcc', '-O3', '-pthread', '-march=native', '-o', bin_path, src_path]
            if extra_flags:
                cmd.extend(extra_flags)
            subprocess.run(cmd, check=True)
        return bin_path

    def train(self, file_paths: List[str], compound_slots=10000, bpe_slots=3264,
              compound_passes=2, template_slots=2000):
        """
        Trains the tokenizer vocabulary from a list of files.
        Multi-pass compound approach:
          Pass 1: Count primitive n-grams (up to 7 atoms), build vocab, encode
          Pass 2..N: Count bigrams of previous pass's tokens to form compound
                     phrases (each pass doubles max reach). Compound budget is
                     split evenly across compounding passes.
        Then BPE on residuals from the final pass.
        Finally, mine 1-slot templates from the token stream.
        """
        print(f"Training tokenizer on {len(file_paths)} files using fast_counter...", flush=True)

        # Determine path to fast_counter binary
        base_dir = os.path.dirname(os.path.abspath(__file__))
        counter_bin = os.path.join(base_dir, "fast_counter")

        if not os.path.exists(counter_bin):
            print("Compiling fast_counter...", flush=True)
            try:
                subprocess.run(
                    ["gcc", "-O3", "-o", counter_bin, os.path.join(base_dir, "fast_counter.c")],
                    check=True
                )
            except Exception as e:
                print(f"Failed to compile fast_counter: {e}", flush=True)
                return

        counts_file = os.path.join(base_dir, "tokenizer", "counts.txt")
        if os.path.exists(counts_file):
            os.remove(counts_file)

        target_file = file_paths[0]

        print(f"Running fast_counter on {target_file}...", flush=True)
        try:
            subprocess.run([counter_bin, target_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"fast_counter failed: {e}", flush=True)
            return

        # Read back the counts
        print("Loading counts from C backend...", flush=True)

        atom_freqs = collections.Counter()
        phrase_freqs = collections.Counter()

        current_section = None

        with open(counts_file, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line == "ATOMS":
                    current_section = "atoms"
                    continue
                elif line in ("BIGRAMS", "TRIGRAMS", "PHRASES"):
                    current_section = "phrases"
                    continue

                parts = line.split(' ')
                if len(parts) < 2: continue

                try:
                    count = int(parts[0])
                    raw_token = " ".join(parts[1:])
                    token = raw_token.replace('\\\\', '\\').replace('\\n', '\n').replace('\\r', '\r')

                    if current_section == "atoms":
                        atom_freqs[token] += count
                    elif current_section == "phrases":
                        phrase_freqs[token] += count
                except ValueError:
                    continue

        print(f"Loaded {len(atom_freqs)} atoms and {len(phrase_freqs)} phrases.", flush=True)

        # ── Budget split ──
        reserved_tokens = [b"<pad>", b"<unk>", b"<s>", b"</s>"]
        reserved_bytes = [bytes([i]) for i in range(256)]
        total_reserved = len(reserved_tokens) + len(reserved_bytes)  # 260
        remaining_slots = self.vocab_size - total_reserved  # 65276

        num_primitive_slots = remaining_slots - compound_slots - template_slots - bpe_slots
        slots_per_pass = compound_slots // compound_passes
        print(f"Budget: {num_primitive_slots} primitive + {compound_slots} compound "
              f"({slots_per_pass}x{compound_passes} passes) + {template_slots} template "
              f"+ {bpe_slots} BPE = {remaining_slots}",
              flush=True)

        # ── Step A: Select primitive phrases ──
        print("Selecting primitive phrases...", flush=True)

        all_phrase_freqs = collections.Counter()
        for atom, freq in atom_freqs.items():
            b = atom.encode('latin-1')
            if len(b) > 1:
                all_phrase_freqs[b] = freq
        for phrase, freq in phrase_freqs.items():
            b = phrase.encode('latin-1')
            all_phrase_freqs[b] += freq

        top_phrases = all_phrase_freqs.most_common(num_primitive_slots)
        primitive_entries = [b for b, _ in top_phrases]

        print(f"  Selected {len(primitive_entries)} primitive phrases.", flush=True)

        # ── Iterative compound passes ──
        current_vocab = reserved_tokens + reserved_bytes + primitive_entries
        all_compound_entries = []

        tmp_vocab_fd, tmp_vocab_path = tempfile.mkstemp(suffix='.vocab')
        os.close(tmp_vocab_fd)
        tmp_tokens_fd, tmp_tokens_path = tempfile.mkstemp(suffix='.tokens')
        os.close(tmp_tokens_fd)

        try:
            encoder_bin = self._ensure_fast_encoder()

            for pass_num in range(1, compound_passes + 1):
                print(f"Pass {pass_num}: encoding with {len(current_vocab)} tokens...", flush=True)
                self._save_vocab(current_vocab, tmp_vocab_path)
                subprocess.run(
                    [encoder_bin, tmp_vocab_path, target_file, tmp_tokens_path],
                    check=True
                )

                token_data = open(tmp_tokens_path, 'rb').read()
                num_tokens = len(token_data) // 2
                print(f"  Pass {pass_num} produced {num_tokens:,} tokens.", flush=True)

                # Count token bigrams — relax min_freq for later passes
                min_freq = max(50, 125 - pass_num * 25)
                print(f"  Counting token bigrams (min_freq={min_freq})...", flush=True)
                bigram_counts = count_token_bigrams(tmp_tokens_path, min_freq=min_freq)
                print(f"  Found {len(bigram_counts):,} bigram types.", flush=True)

                # Build compound phrases from bigrams
                compound_candidates = []
                for (id_a, id_b), freq in bigram_counts.items():
                    if id_a >= len(current_vocab) or id_b >= len(current_vocab):
                        continue
                    compound = current_vocab[id_a] + current_vocab[id_b]
                    if len(compound) <= 1:
                        continue
                    score = freq * len(compound)
                    compound_candidates.append((score, compound))

                compound_candidates.sort(reverse=True)

                # Dedup against everything already in vocab
                existing = set(current_vocab)
                new_compounds = []
                for _, compound in compound_candidates:
                    if compound in existing:
                        continue
                    existing.add(compound)
                    new_compounds.append(compound)
                    if len(new_compounds) >= slots_per_pass:
                        break

                print(f"  Selected {len(new_compounds)} compound phrases (pass {pass_num}).", flush=True)
                all_compound_entries.extend(new_compounds)
                current_vocab = reserved_tokens + reserved_bytes + primitive_entries + all_compound_entries

            # ── Final encode to collect residuals ──
            print(f"Final encode: {len(current_vocab)} tokens...", flush=True)
            self._save_vocab(current_vocab, tmp_vocab_path)
            subprocess.run(
                [encoder_bin, tmp_vocab_path, target_file, tmp_tokens_path],
                check=True
            )

            token_data = open(tmp_tokens_path, 'rb').read()
            num_tokens = len(token_data) // 2
            print(f"  Final encode produced {num_tokens:,} tokens.", flush=True)

            # Collect residuals
            residuals = []
            current_run = bytearray()

            for offset in range(0, len(token_data), 2):
                tid = int.from_bytes(token_data[offset:offset+2], 'little')
                if 4 <= tid <= 259:
                    current_run.append(tid - 4)
                else:
                    if len(current_run) > 1:
                        residuals.append(bytes(current_run))
                    current_run = bytearray()

            if len(current_run) > 1:
                residuals.append(bytes(current_run))

            print(f"  Collected {len(residuals):,} residual segments "
                  f"({sum(len(r) for r in residuals):,} bytes total).", flush=True)

        finally:
            os.unlink(tmp_vocab_path)
            os.unlink(tmp_tokens_path)

        # ── Train BPE on residuals ──
        print(f"Training BPE on residuals ({bpe_slots} merges)...", flush=True)
        bpe_tokens = train_bpe(residuals, bpe_slots)

        existing = set(current_vocab)
        bpe_tokens = [t for t in bpe_tokens if t not in existing and len(t) < 4096]
        print(f"  Got {len(bpe_tokens)} unique BPE tokens.", flush=True)

        # ── Build interim vocab (pre-template) ──
        interim_vocab = current_vocab + bpe_tokens
        if len(interim_vocab) > self.vocab_size - template_slots:
            interim_vocab = interim_vocab[:self.vocab_size - template_slots]

        # ── Mine word-level templates using fast_mask + fast_ngram ──
        self.templates = {}
        num_templates_found = 0
        if template_slots > 0:
            print(f"\n── Template mining (word-level, pre-encoding) ──", flush=True)
            MASK_SENTINEL_ID = 0xFFFFFFFF
            base_dir = os.path.dirname(os.path.abspath(__file__))
            mask_bin = self._ensure_c_binary('fast_mask')
            ngram_bin = self._ensure_c_binary('fast_ngram')

            words_bin = os.path.join(base_dir, 'data', 'masked_words.bin')
            vocab_bin = os.path.join(base_dir, 'data', 'mask_vocab.bin')
            ngrams_bin = os.path.join(base_dir, 'data', 'ngrams.bin')

            # Step 1: fast_mask — text → masked word stream + word vocab
            print(f"  Running fast_mask...", flush=True)
            result = subprocess.run([mask_bin, target_file, words_bin, vocab_bin],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"fast_mask failed: {result.stderr}")
            print(f"  {result.stderr.strip()}", flush=True)

            # Read word vocab
            data = open(vocab_bin, 'rb').read()
            n_words = int.from_bytes(data[0:4], 'little')
            off = 4
            id2word = {}
            for i in range(n_words):
                wlen = int.from_bytes(data[off:off+4], 'little')
                off += 4
                id2word[i] = data[off:off+wlen].decode('utf-8', errors='replace')
                off += wlen

            # Step 2: fast_ngram count — word-level n-grams
            n_threads = min(os.cpu_count() or 4, 6)
            print(f"  Running fast_ngram count (n=4..7, min_freq=100)...", flush=True)
            result = subprocess.run(
                [ngram_bin, 'count', words_bin, ngrams_bin,
                 '4', '7', '100', str(n_threads)],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"fast_ngram count failed: {result.stderr}")
            print(f"  {result.stderr.strip()}", flush=True)

            # Read binary n-gram output
            with open(ngrams_bin, 'rb') as f:
                hdr = np.fromfile(f, dtype='<u4', count=2)
                num_entries, max_n_out = int(hdr[0]), int(hdr[1])
                dt = np.dtype([
                    ('count', '<u4'), ('n_len', 'u1'), ('has_sent', 'u1'),
                    ('_pad', '<u2'), ('ids', '<u4', (max_n_out,)),
                ])
                entries = np.fromfile(f, dtype=dt, count=num_entries)
            print(f"  {num_entries:,} n-gram entries", flush=True)

            # Step 3: Score word-level templates directly (no token conversion)
            candidates = []
            n_skipped = 0
            for e in entries:
                if not e['has_sent']:
                    continue
                n_len = int(e['n_len'])
                count = int(e['count'])
                word_ids = [int(e['ids'][j]) for j in range(n_len)]
                n_slots = sum(1 for x in word_ids if x == MASK_SENTINEL_ID)
                if n_slots != 1:
                    continue

                # Find slot position and resolve word strings
                word_slot_pos = -1
                words = []
                valid = True
                for i, wid in enumerate(word_ids):
                    if wid == MASK_SENTINEL_ID:
                        word_slot_pos = i
                        words.append(None)  # slot placeholder
                    else:
                        w = id2word.get(wid)
                        if w is None:
                            valid = False
                            break
                        words.append(w)
                if not valid:
                    n_skipped += 1
                    continue

                # Score: each match saves (num_words - 2) tokens
                # (entire span → template_id + fill = 2 tokens, was num_words tokens)
                saved = n_len - 2
                if saved < 1:
                    continue
                score = saved * count
                # Dedup key: tuple of word strings (None for slot)
                key = tuple(words)
                candidates.append((score, key, word_slot_pos, count, words))

            candidates.sort(key=lambda x: -x[0])
            print(f"  {len(candidates):,} template candidates "
                  f"({n_skipped:,} skipped: word lookup failed)", flush=True)

            # Dedup by word pattern, select top
            seen = set()
            raw_templates = []
            for score, key, slot_pos, count, words in candidates:
                if key in seen:
                    continue
                seen.add(key)
                raw_templates.append((words, slot_pos, score, count))
                if len(raw_templates) >= template_slots:
                    break

            # Assign template IDs and build TemplateInfo with decode bytes
            base_id = len(interim_vocab)
            for i, (words, slot_pos, score, count) in enumerate(raw_templates):
                vid = base_id + i
                # Build bytes_before_fill / bytes_after_fill for decoding
                before_words = [w for w in words[:slot_pos] if w is not None]
                after_words = [w for w in words[slot_pos + 1:] if w is not None]
                # In text, words are separated by spaces. Template matches
                # include the inter-word whitespace. The first word in the
                # match doesn't have leading whitespace (it's part of the
                # preceding trie encoding region).
                if slot_pos > 0:
                    # before_fill = " ".join(before_words) as bytes
                    # Leading space on first before-word is NOT included
                    # (it was consumed by preceding trie encoding).
                    # But inter-word spaces between before-words ARE included.
                    before_str = ' '.join(before_words)
                    bytes_before = before_str.encode('utf-8')
                else:
                    bytes_before = b""
                if slot_pos < len(words) - 1:
                    after_str = ' '.join(after_words)
                    bytes_after = (' ' + after_str).encode('utf-8')
                else:
                    bytes_after = b""

                # Word strings for the template (None at slot position)
                word_strings = list(words)

                tmpl = TemplateInfo(
                    vocab_id=vid,
                    num_words=len(words),
                    slot_pos=slot_pos,
                    word_strings=word_strings,
                    bytes_before_fill=bytes_before,
                    bytes_after_fill=bytes_after,
                )
                self.templates[vid] = tmpl

            num_templates_found = len(self.templates)
            if num_templates_found > 0:
                print(f"\n  Top 20 templates:", flush=True)
                for i, (words, slot_pos, score, count) in enumerate(raw_templates[:20]):
                    display = [w if w is not None else ';?' for w in words]
                    print(f"    {i+1:3d}. score={score:>10,}  freq={count:>6,}  "
                          f"nW={len(words)}  slot={slot_pos}  "
                          f"words=[{' '.join(display)}]", flush=True)

            print(f"  Selected {num_templates_found} templates.", flush=True)

        # ── Build final vocab ──
        self.vocab = interim_vocab
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self._build_trie()

        print(f"\nVocab size: {len(self.vocab)} + {num_templates_found} templates", flush=True)
        print(f"  - Special: {len(reserved_tokens)}", flush=True)
        print(f"  - Bytes: {len(reserved_bytes)}", flush=True)
        print(f"  - Primitives: {len(primitive_entries)}", flush=True)
        print(f"  - Compounds: {len(all_compound_entries)} ({compound_passes} passes)", flush=True)
        print(f"  - BPE: {len(bpe_tokens)}", flush=True)
        print(f"  - Templates: {num_templates_found}", flush=True)
        self.save(self.model_path)

    def _build_trie(self):
        """
        Builds a trie for fast greedy longest-match.
        """
        self.trie = {}
        for i, token_bytes in enumerate(self.vocab):
            node = self.trie
            for byte in token_bytes:
                if byte not in node:
                    node[byte] = {}
                node = node[byte]
            node['__id__'] = i

    def encode(self, text: str) -> List[int]:
        """
        Greedy encoding: always takes the longest matching token from the current position.
        """
        # ... (keep existing implementation for small strings) ...
        text_bytes = text.encode('utf-8')
        ids = []
        i = 0
        n = len(text_bytes)
        
        unk_id = self.token_to_id.get(b"<unk>", 1)
        
        while i < n:
            node = self.trie
            longest_match_id = -1
            longest_match_len = 0
            
            j = i
            while j < n:
                byte = text_bytes[j]
                if byte in node:
                    node = node[byte]
                    if '__id__' in node:
                        longest_match_id = node['__id__']
                        longest_match_len = (j - i) + 1
                    j += 1
                else:
                    break
            
            if longest_match_id != -1:
                ids.append(longest_match_id)
                i += longest_match_len
            else:
                byte_val = bytes([text_bytes[i]])
                byte_id = self.token_to_id.get(byte_val)
                if byte_id is not None:
                    ids.append(byte_id)
                else:
                    ids.append(unk_id)
                i += 1
                
        return ids

    def encode_file(self, input_path: str, output_path: str,
                    output_meta_path: Optional[str] = None):
        """
        Encodes a large file using the C fast_encoder binary.
        If templates are loaded, passes the .templates file as 4th arg
        for integrated word-level template matching during encoding.
        """
        print(f"Encoding {input_path} to {output_path} using fast_encoder...", flush=True)
        encoder_bin = self._ensure_fast_encoder()

        # Ensure vocab is saved for C to read
        if not os.path.exists(self.model_path):
            self.save(self.model_path)

        cmd = [encoder_bin, self.model_path, input_path, output_path]

        # Pass templates file as optional 4th arg for hybrid encoding
        if self.templates:
            tmpl_path = self.model_path.replace('.vocab', '.templates') if '.vocab' in self.model_path else self.model_path + '.templates'
            if not os.path.exists(tmpl_path):
                self.save(self.model_path)
            cmd.append(tmpl_path)

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"fast_encoder failed: {e}", flush=True)
            raise

    def decode(self, ids: List[int]) -> str:
        res = []
        i = 0
        while i < len(ids):
            tid = ids[i]
            if tid in self.templates:
                tmpl = self.templates[tid]
                # Next token is the fill
                fill_tid = ids[i + 1] if i + 1 < len(ids) else 0
                res.append(tmpl.bytes_before_fill)
                if 0 <= fill_tid < len(self.vocab):
                    res.append(self.vocab[fill_tid])
                res.append(tmpl.bytes_after_fill)
                i += 2  # template_id + fill
            elif 0 <= tid < len(self.vocab):
                res.append(self.vocab[tid])
                i += 1
            else:
                i += 1  # skip invalid IDs
        return b"".join(res).decode('utf-8', errors='replace')

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('>I', len(self.vocab)))
            for token in self.vocab:
                f.write(struct.pack('>I', len(token)))
                f.write(token)

        # Save word-level templates companion file
        if self.templates:
            tmpl_path = path.replace('.vocab', '.templates') if '.vocab' in path else path + '.templates'
            sorted_tmpls = sorted(self.templates.values(), key=lambda t: t.vocab_id)
            base_id = sorted_tmpls[0].vocab_id if sorted_tmpls else 0
            with open(tmpl_path, 'wb') as f:
                f.write(struct.pack('>I', len(sorted_tmpls)))
                f.write(struct.pack('>I', base_id))
                for tmpl in sorted_tmpls:
                    f.write(struct.pack('BB', tmpl.num_words, tmpl.slot_pos))
                    for j, word in enumerate(tmpl.word_strings):
                        if j == tmpl.slot_pos:
                            continue  # skip the slot position
                        word_bytes = word.encode('utf-8')
                        f.write(struct.pack('>H', len(word_bytes)))
                        f.write(word_bytes)
            print(f"Saved {len(sorted_tmpls)} word-templates to {tmpl_path}", flush=True)
                
    def load(self, path):
        with open(path, 'rb') as f:
            vocab_len = struct.unpack('>I', f.read(4))[0]
            self.vocab = []
            for _ in range(vocab_len):
                token_len = struct.unpack('>I', f.read(4))[0]
                token = f.read(token_len)
                self.vocab.append(token)

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self._build_trie()

        # Load word-level templates companion file if it exists
        tmpl_path = path.replace('.vocab', '.templates') if '.vocab' in path else path + '.templates'
        self.templates = {}
        if os.path.exists(tmpl_path):
            with open(tmpl_path, 'rb') as f:
                num_templates = struct.unpack('>I', f.read(4))[0]
                base_id = struct.unpack('>I', f.read(4))[0]
                for i in range(num_templates):
                    num_words, slot_pos = struct.unpack('BB', f.read(2))
                    word_strings = []
                    for j in range(num_words):
                        if j == slot_pos:
                            word_strings.append(None)
                            continue
                        wlen = struct.unpack('>H', f.read(2))[0]
                        word_bytes = f.read(wlen)
                        word_strings.append(word_bytes.decode('utf-8'))
                    vid = base_id + i
                    # Derive bytes_before_fill / bytes_after_fill
                    before_words = [w for w in word_strings[:slot_pos] if w is not None]
                    after_words = [w for w in word_strings[slot_pos + 1:] if w is not None]
                    if slot_pos > 0:
                        bytes_before = ' '.join(before_words).encode('utf-8')
                    else:
                        bytes_before = b""
                    if slot_pos < num_words - 1:
                        bytes_after = (' ' + ' '.join(after_words)).encode('utf-8')
                    else:
                        bytes_after = b""
                    self.templates[vid] = TemplateInfo(
                        vocab_id=vid,
                        num_words=num_words,
                        slot_pos=slot_pos,
                        word_strings=word_strings,
                        bytes_before_fill=bytes_before,
                        bytes_after_fill=bytes_after,
                    )
            print(f"Loaded {len(self.templates)} word-templates from {tmpl_path}", flush=True)

if __name__ == "__main__":
    # Demo
    t = GreedyPhraseTokenizer(vocab_size=1000) # Small vocab for demo
    
    # Create a dummy training file
    with open("tokenizer/train_dummy.txt", "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 10)
        f.write("guest appearance " * 20)
        f.write("Hello world. " * 10)
    
    t.train(["tokenizer/train_dummy.txt"], compound_slots=100, bpe_slots=100, template_slots=0)
    
    text = "The guest appearance was quick."
    tokens = t.encode(text)
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{t.decode(tokens)}'")
    
    # Verify "guest appearance" is one token
    ga_id = t.token_to_id.get(b"guest appearance")
    if ga_id and ga_id in tokens:
        print("[PASS] 'guest appearance' is a single token!")
    else:
        print("[FAIL] 'guest appearance' was split.")
