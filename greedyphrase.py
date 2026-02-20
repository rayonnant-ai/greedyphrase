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
    """Metadata for a single 1-slot template pattern."""
    vocab_id: int              # assigned ID in the vocab (>= len(base_vocab))
    length: int                # number of token positions in the template
    num_slots: int = 1         # always 1 for phase 6
    frame: List[int] = field(default_factory=list)  # token IDs with SLOT_SENTINEL at slot positions
    slot_positions: List[int] = field(default_factory=list)  # indices where slots appear
    frame_bytes: List[bytes] = field(default_factory=list) # bytes of the tokens in the frame (for re-mapping)


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


def count_token_ngrams(tokens_path, max_n=7, min_freq=100):
    """Count token n-grams for n=2..max_n from a .tokens file using numpy.

    n=2..4: pack n uint16 IDs into one uint64 via bit-shifting, use np.unique.
    n=5..7: pack into structured dtype (hi: uint64, lo: uint64), use np.unique.
    Returns dict[tuple[int,...], int] filtered by min_freq.
    """
    tokens = np.fromfile(tokens_path, dtype=np.uint16)
    result = {}
    N = len(tokens)

    for n in range(2, max_n + 1):
        if N < n:
            continue

        if n <= 4:
            # Pack n token IDs into a single uint64
            packed = np.zeros(N - n + 1, dtype=np.uint64)
            for i in range(n):
                packed |= tokens[i:N - n + 1 + i].astype(np.uint64) << np.uint64((n - 1 - i) * 16)
            unique_packed, counts = np.unique(packed, return_counts=True)
            mask = counts >= min_freq
            for p, c in zip(unique_packed[mask], counts[mask]):
                ids = tuple(int((p >> np.uint64((n - 1 - i) * 16)) & np.uint64(0xFFFF)) for i in range(n))
                result[ids] = int(c)
        else:
            # n=5..7: pack into (hi: uint64, lo: uint64)
            # hi = first 4 tokens, lo = remaining tokens
            hi = np.zeros(N - n + 1, dtype=np.uint64)
            for i in range(4):
                hi |= tokens[i:N - n + 1 + i].astype(np.uint64) << np.uint64((3 - i) * 16)
            lo = np.zeros(N - n + 1, dtype=np.uint64)
            rem = n - 4
            for i in range(rem):
                lo |= tokens[4 + i:N - n + 1 + 4 + i].astype(np.uint64) << np.uint64((rem - 1 - i) * 16)

            dt = np.dtype([('hi', np.uint64), ('lo', np.uint64)])
            packed = np.empty(N - n + 1, dtype=dt)
            packed['hi'] = hi
            packed['lo'] = lo
            unique_packed, counts = np.unique(packed, return_counts=True)
            mask = counts >= min_freq
            for p, c in zip(unique_packed[mask], counts[mask]):
                h, l = int(p['hi']), int(p['lo'])
                ids = tuple(int((h >> ((3 - i) * 16)) & 0xFFFF) for i in range(4))
                ids += tuple(int((l >> ((rem - 1 - i) * 16)) & 0xFFFF) for i in range(rem))
                result[ids] = int(c)

    return result


# ─── Word-level n-gram counting (for hollow mining on fast_mask output) ───

_MASK_SENTINEL = np.uint32(0xFFFFFFFF)
_MASK_LINE_SEP = np.uint32(0xFFFFFFFE)


def count_word_ngrams(word_ids, max_n=7, min_freq=100):
    """Count n-grams in a uint32 word-ID stream from fast_mask.

    Skips n-grams containing SENTINEL (0xFFFFFFFF) or LINE_SEP (0xFFFFFFFE).
    Returns dict[tuple[int,...], int] filtered by min_freq.
    """
    N = len(word_ids)
    result = {}
    bad = word_ids >= _MASK_LINE_SEP  # both SENTINEL and LINE_SEP

    for n in range(2, max_n + 1):
        if N < n:
            continue
        M = N - n + 1

        # Valid positions: no bad ID in the window
        valid = np.ones(M, dtype=bool)
        for j in range(n):
            valid &= ~bad[j:j + M]

        nvalid = valid.sum()
        if nvalid == 0:
            continue

        # Pack into structured dtype
        valid_idx = np.where(valid)[0]
        dt = np.dtype([(f'w{j}', np.uint32) for j in range(n)])
        packed = np.empty(nvalid, dtype=dt)
        for j in range(n):
            packed[f'w{j}'] = word_ids[valid_idx + j]

        unique, counts = np.unique(packed, return_counts=True)
        mask = counts >= min_freq
        for p, c in zip(unique[mask], counts[mask]):
            ids = tuple(int(p[f'w{j}']) for j in range(n))
            result[ids] = int(c)

    return result


def replace_word_ngrams(stream, ngrams, start_id):
    """Replace matched n-grams in a uint32 stream with compound IDs.

    Numpy-vectorized, greedy longest-first left-to-right.
    Uses polynomial hashing for fast candidate detection, then greedy assignment.
    Returns (new_stream_array, compound_map: {id: ngram_tuple}).
    """
    N = len(stream)
    compound_map = {}
    HASH_PRIME = np.uint64(2654435761)

    # Group by length, assign CIDs
    by_length = collections.defaultdict(list)
    for i, ngram in enumerate(ngrams):
        cid = start_id + i
        by_length[len(ngram)].append((ngram, cid))
        compound_map[cid] = ngram

    # Track replacements
    is_start = np.zeros(N, dtype=bool)
    cid_at = np.zeros(N, dtype=np.uint32)
    len_at = np.zeros(N, dtype=np.int32)
    consumed = np.zeros(N, dtype=bool)

    # Process longest first (greedy)
    for L in sorted(by_length.keys(), reverse=True):
        targets = by_length[L]
        M = N - L + 1
        if M <= 0:
            continue

        # Compute polynomial hash for each window (vectorized numpy)
        window_hash = np.zeros(M, dtype=np.uint64)
        for j in range(L):
            window_hash = window_hash * HASH_PRIME + stream[j:j + M].astype(np.uint64)

        # Hash targets + build lookup
        target_hash_to_cid = {}
        target_hash_list = []
        for ngram, cid in targets:
            h = np.uint64(0)
            for v in ngram:
                h = h * HASH_PRIME + np.uint64(v)
            target_hash_to_cid[int(h)] = cid
            target_hash_list.append(int(h))

        target_hash_arr = np.unique(np.array(target_hash_list, dtype=np.uint64))
        if len(target_hash_arr) == 0:
            continue

        # Find candidate positions: hash match AND not consumed
        candidate_mask = np.isin(window_hash, target_hash_arr)
        for j in range(L):
            candidate_mask &= ~consumed[j:j + M]

        candidate_positions = np.where(candidate_mask)[0]
        if len(candidate_positions) == 0:
            continue

        # Pre-extract hashes as Python list for fast iteration
        candidate_hashes = window_hash[candidate_positions].tolist()
        candidate_positions = candidate_positions.tolist()

        # Greedy left-to-right assignment
        next_free = 0
        for pos, h in zip(candidate_positions, candidate_hashes):
            if pos < next_free:
                continue
            cid = target_hash_to_cid.get(h)
            if cid is not None:
                is_start[pos] = True
                cid_at[pos] = cid
                len_at[pos] = L
                consumed[pos:pos + L] = True
                next_free = pos + L

    # Build output: replace match starts with CID, skip consumed tails
    sel = np.where(is_start)[0]
    is_skip = np.zeros(N, dtype=bool)
    if len(sel) > 0:
        sel_lens = len_at[sel]
        skip_counts = sel_lens - 1
        total_skip = int(skip_counts.sum())
        if total_skip > 0:
            offsets = np.concatenate([np.arange(1, L) for L in sel_lens])
            bases = np.repeat(sel, skip_counts)
            is_skip[bases + offsets] = True

    output = np.where(is_start, cid_at, stream)
    output = output[~is_skip]

    return output, compound_map


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
    def __init__(self, vocab_size=65536, model_path="tokenizer/greedyphrase.vocab",
                 event_model=False):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.vocab: List[bytes] = []
        self.token_to_id: Dict[bytes, int] = {}
        self.trie = {} # Character-based trie for fast greedy matching
        self.templates: Dict[int, TemplateInfo] = {}  # vocab_id -> TemplateInfo
        self.event_model = event_model
        self._event_parser = None
        if event_model:
            from event_parser import EventParser
            self._event_parser = EventParser()

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

    def _ensure_miner(self):
        """Compile fast_mask and fast_template_miner if needed."""
        self._ensure_c_binary("fast_mask")
        self._ensure_c_binary("fast_template_miner")

    def _mine_linguistic_templates(self, target_file, budget=2000, min_freq=100):
        """Mine templates using the fast C pipeline (Multi-pass)."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        masked_words = os.path.join(data_dir, "masked_words.bin")
        mask_vocab = os.path.join(data_dir, "mask_vocab.bin")
        winning_templates = os.path.join(data_dir, "winning_templates.txt")
        
        self._ensure_miner()
        
        print(f"  Step 1: Masking {target_file}...", flush=True)
        mask_bin = os.path.join(base_dir, "fast_mask")
        subprocess.run([mask_bin, target_file, masked_words, mask_vocab], check=True)
        
        print(f"  Step 2: Mining n-grams from masked stream (Multi-pass)...", flush=True)
        miner_bin = os.path.join(base_dir, "fast_template_miner")
        
        # Split mining into 3 passes to avoid OOM with large thread count
        # Pass 1: 5..8, Pass 2: 9..12, Pass 3: 13..15
        passes = [(5, 8), (9, 12), (13, 15)]
        all_templates = []
        
        for i, (min_n, max_n) in enumerate(passes):
            print(f"    Pass {i+1}: Mining L={min_n}..{max_n}...", flush=True)
            raw_templates_part = os.path.join(data_dir, f"raw_templates_p{i+1}.txt")
            if os.path.exists(raw_templates_part): os.remove(raw_templates_part)
            
            subprocess.run([
                miner_bin, masked_words, mask_vocab, raw_templates_part, 
                str(min_freq), str(min_n), str(max_n), "12"
            ], check=True)
            
            # Load and merge results immediately to save disk/memory
            with open(raw_templates_part, 'r', encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 2: continue
                    try:
                        count = int(parts[0])
                        tokens = parts[1:]
                        num_slots = tokens.count(';?')
                        length = len(tokens)
                        saved = length - (1 + num_slots)
                        if saved <= 0: continue
                        score = count * saved
                        all_templates.append((score, count, tokens))
                    except ValueError: continue
            
            os.remove(raw_templates_part) # Cleanup
        
        print(f"  Step 3: Promoting top {budget} templates from {len(all_templates)} candidates...", flush=True)
        all_templates.sort(key=lambda x: -x[0])
        winners = all_templates[:budget]
        
        # Save winners for inspection
        with open(winning_templates, 'w', encoding='latin-1') as f:
            for score, count, tokens in winners:
                f.write(f"{count} {' '.join(tokens)}\n")
        
        return winners

    def _mine_hollow_compounds(self, text_path, max_n=7, min_freq=100):
        """Iterative hollow compound mining using fast_mask.

        Pass 1: fast_mask → count word n-grams in the uint32 stream.
        Pass 2+: replace discovered n-grams with compound IDs, mask all
                 remaining original word IDs → count again.
        Until convergence (<1% stream shrink) or 20 passes.

        Returns list of (freq, compound_bytes) for the candidate pool.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        mask_bin = self._ensure_c_binary("fast_mask")
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        words_path = os.path.join(data_dir, "mask_words.bin")
        vocab_path = os.path.join(data_dir, "mask_vocab.bin")

        print("  Running fast_mask...", flush=True)
        subprocess.run([mask_bin, text_path, words_path, vocab_path], check=True)

        # Load word vocab
        word_vocab = []
        with open(vocab_path, 'rb') as f:
            n_words = struct.unpack('<I', f.read(4))[0]
            for _ in range(n_words):
                wlen = struct.unpack('<I', f.read(4))[0]
                word_vocab.append(f.read(wlen))

        stream = np.fromfile(words_path, dtype=np.uint32)
        n_word_ids = len(word_vocab)  # original word IDs are 0..n_word_ids-1

        all_compound_map = {}  # compound_id -> word_id tuple
        next_cid = n_word_ids
        prev_len = len(stream)

        for pass_num in range(1, 21):
            print(f"  Hollow pass {pass_num}: stream {len(stream):,} tokens", flush=True)

            ngram_counts = count_word_ngrams(stream, max_n=max_n, min_freq=min_freq)
            if not ngram_counts:
                print(f"  No frequent n-grams found, stopping.", flush=True)
                break

            # Sort by (-freq, -len) — prefer longer at same frequency
            sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: (-x[1], -len(x[0])))

            # Take top 10K that aren't already discovered
            known = set(all_compound_map.values())
            new_ngrams = []
            for ngram, freq in sorted_ngrams:
                if ngram in known:
                    continue
                new_ngrams.append(ngram)
                if len(new_ngrams) >= 10000:
                    break

            if not new_ngrams:
                print(f"  No new n-grams, stopping.", flush=True)
                break

            print(f"  Replacing {len(new_ngrams)} n-grams...", flush=True)
            stream, new_map = replace_word_ngrams(stream, new_ngrams, next_cid)
            all_compound_map.update(new_map)
            next_cid += len(new_ngrams)

            # Mask: replace all original word IDs (not compounds, not special) with SENTINEL
            is_original = stream < n_word_ids
            stream[is_original] = _MASK_SENTINEL

            new_len = len(stream)
            drop_pct = 100.0 * (prev_len - new_len) / prev_len if prev_len > 0 else 0
            print(f"  Stream: {prev_len:,} -> {new_len:,} (shrink {drop_pct:.1f}%)", flush=True)
            if drop_pct < 1.0 and pass_num > 1:
                print(f"  Converged.", flush=True)
                break
            prev_len = new_len

        # Resolve compound IDs back to byte strings
        def resolve(wid):
            if wid in all_compound_map:
                parts = []
                for sub in all_compound_map[wid]:
                    parts.extend(resolve(sub))
                return parts
            elif wid < n_word_ids:
                return [word_vocab[wid]]
            else:
                return []  # SENTINEL — shouldn't appear in compounds

        # Count all IDs in one pass (O(N log N) instead of O(K * N))
        unique_ids, unique_counts = np.unique(stream, return_counts=True)
        freq_map = dict(zip(unique_ids.astype(int), unique_counts.astype(int)))

        results = []
        for cid in all_compound_map:
            word_parts = resolve(cid)
            if not word_parts:
                continue
            compound_bytes = b' '.join(word_parts)
            freq = freq_map.get(cid, 0)
            if freq > 0:
                results.append((freq, compound_bytes))

        print(f"  Hollow mining: {len(results)} compounds discovered", flush=True)
        return results

    def train(self, file_paths: List[str], max_vocab_size=65536,
              compound_passes=2, compound_max_n=2, compound_converge=False,
              hollow=False, bpe_slots=3000, template_budget=2000):
        """
        Trains the tokenizer vocabulary from a list of files using "Free Market" selection.
        """
        print(f"Training tokenizer on {len(file_paths)} files (Free Market Mode)...", flush=True)
        target_file = file_paths[0]

        # Event model preprocessing: transform corpus before training
        if self.event_model:
            event_file = target_file + "_event.txt"
            if not os.path.exists(event_file):
                print(f"Transforming corpus to event model text...", flush=True)
                text = open(target_file, 'r', encoding='utf-8', errors='replace').read()
                event_text = self._event_parser.transform(text)
                with open(event_file, 'w', encoding='utf-8') as f:
                    f.write(event_text)
                print(f"  Event text: {len(event_text):,} bytes "
                      f"({len(event_text)/len(text)*100:.1f}% of original)", flush=True)
            else:
                print(f"Using cached event model text: {event_file}", flush=True)
            target_file = event_file
            file_paths = [event_file]

        # ── New Step 0: Mine Linguistic Templates ──
        winning_templates_data = []
        if template_budget > 0:
            print("Mining linguistic templates (Phase 7C)...", flush=True)
            winning_templates_data = self._mine_linguistic_templates(target_file, budget=template_budget)
        else:
            print("Skipping linguistic template mining (budget=0)...", flush=True)

        # Determine path to fast_counter binary
        base_dir = os.path.dirname(os.path.abspath(__file__))
        counter_bin = self._ensure_c_binary("fast_counter")

        counts_file = os.path.join(base_dir, "tokenizer", "counts.txt")
        if os.path.exists(counts_file):
            os.remove(counts_file)

        print(f"Running fast_counter on {target_file}...", flush=True)
        subprocess.run([counter_bin, target_file], check=True)

        # Read back the counts
        print("Loading counts from C backend...", flush=True)
        atom_freqs = collections.Counter()
        phrase_freqs = collections.Counter()
        current_section = None
        with open(counts_file, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line == "ATOMS": current_section = "atoms"
                elif line in ("BIGRAMS", "TRIGRAMS", "PHRASES"): current_section = "phrases"
                else:
                    parts = line.split(' ')
                    if len(parts) < 2: continue
                    try:
                        count = int(parts[0])
                        token = " ".join(parts[1:]).replace('\\n', '\n').replace('\\r', '\r')
                        if current_section == "atoms": atom_freqs[token] += count
                        elif current_section == "phrases": phrase_freqs[token] += count
                    except ValueError: continue

        # ── Candidate Pool ──
        # type 0: Token (bytes), type 1: Template (TemplateInfo)
        candidate_pool = []
        
        # Add mined linguistic templates to pool
        template_candidates = []
        for score, count, words in winning_templates_data:
            # We need to store the word list to reconstruct the TemplateInfo later
            candidate_pool.append((count, 1, words))
            
            # Also add literal words to pool with high frequency to ensure they exist
            for w in words:
                if w != ';?':
                    candidate_pool.append((1e12, 0, w.encode('latin-1')))

        # Collect Primitives from fast_counter
        all_phrase_freqs = collections.Counter()
        for atom, freq in atom_freqs.items():
            b = atom.encode('latin-1')
            if len(b) > 1: all_phrase_freqs[b] = freq
        for phrase, freq in phrase_freqs.items():
            b = phrase.encode('latin-1')
            all_phrase_freqs[b] += freq

        top_phrases = all_phrase_freqs.most_common(60000)
        for b, freq in top_phrases:
            candidate_pool.append((freq, 0, b))

        reserved_tokens = [b"<pad>", b"<unk>", b"<s>", b"</s>"]
        reserved_bytes = [bytes([i]) for i in range(256)]
        bootstrap_primitives = [b for b, _ in top_phrases[:50000]]
        current_vocab = reserved_tokens + reserved_bytes + bootstrap_primitives

        # Step B: Collect Compounds
        tmp_vocab_fd, tmp_vocab_path = tempfile.mkstemp(suffix='.vocab')
        os.close(tmp_vocab_fd)
        tmp_tokens_fd, tmp_tokens_path = tempfile.mkstemp(suffix='.tokens')
        os.close(tmp_tokens_fd)

        try:
            encoder_bin = self._ensure_fast_encoder()

            if hollow:
                # ── Hollow mining: mask → count word n-grams → mask non-ngram → repeat ──
                print("Hollow compound mining...", flush=True)
                hollow_compounds = self._mine_hollow_compounds(
                    target_file, max_n=compound_max_n, min_freq=100)
                for freq, compound_bytes in hollow_compounds:
                    candidate_pool.append((freq, 0, compound_bytes))
                    current_vocab.append(compound_bytes)
            else:
                # ── Standard compound loop: encode → count token n-grams → repeat ──
                max_passes = 20 if compound_converge else compound_passes
                pass_num = 0
                prev_token_count = None
                while True:
                    pass_num += 1
                    if pass_num > max_passes:
                        break
                    print(f"Pass {pass_num}: encoding with {len(current_vocab)} tokens...", flush=True)
                    self._save_vocab(current_vocab, tmp_vocab_path)
                    subprocess.run([encoder_bin, tmp_vocab_path, target_file, tmp_tokens_path], check=True)

                    token_count = os.path.getsize(tmp_tokens_path) // 2
                    if compound_converge and prev_token_count is not None:
                        drop_pct = 100.0 * (prev_token_count - token_count) / prev_token_count
                        print(f"  Token count: {token_count:,} (drop {drop_pct:.2f}%)", flush=True)
                        if drop_pct < 1.0:
                            print(f"  Converged (<1% drop), stopping.", flush=True)
                            break
                    else:
                        print(f"  Token count: {token_count:,}", flush=True)
                    prev_token_count = token_count

                    if compound_max_n > 2:
                        ngram_counts = count_token_ngrams(tmp_tokens_path, max_n=compound_max_n,
                                                          min_freq=max(50, 125 - pass_num * 25))
                        sorted_ngrams = sorted(ngram_counts.items(), key=lambda x: (-x[1], -len(x[0])))
                    else:
                        bigram_counts = count_token_bigrams(tmp_tokens_path, min_freq=max(50, 125 - pass_num * 25))
                        sorted_ngrams = sorted(bigram_counts.items(), key=lambda x: -x[1])

                    new_compounds = []
                    existing = set(current_vocab)
                    for ids, freq in sorted_ngrams:
                        if any(i >= len(current_vocab) for i in ids):
                            continue
                        compound = b"".join(current_vocab[i] for i in ids)
                        if len(compound) > 1 and compound not in existing:
                            candidate_pool.append((freq, 0, compound))
                            new_compounds.append(compound)
                            existing.add(compound)
                            if len(new_compounds) >= 10000:
                                break
                    if not new_compounds:
                        print(f"  No new compounds found, stopping.", flush=True)
                        break
                    current_vocab.extend(new_compounds)
                    if not compound_converge and pass_num >= compound_passes:
                        break

            # Step C: BPE
            print(f"Final encode for BPE...", flush=True)
            self._save_vocab(current_vocab, tmp_vocab_path)
            subprocess.run([encoder_bin, tmp_vocab_path, target_file, tmp_tokens_path], check=True)
            token_data = open(tmp_tokens_path, 'rb').read()
            residuals = []
            current_run = bytearray()
            for offset in range(0, len(token_data), 2):
                tid = int.from_bytes(token_data[offset:offset+2], 'little')
                if 4 <= tid <= 259: current_run.append(tid - 4)
                else:
                    if len(current_run) > 1: residuals.append(bytes(current_run))
                    current_run = bytearray()
            if len(current_run) > 1: residuals.append(bytes(current_run))
            bpe_tokens = train_bpe(residuals, bpe_slots)

        finally:
            os.unlink(tmp_vocab_path)
            os.unlink(tmp_tokens_path)

        # ── Step E: The Grand Cull ──
        candidate_pool.sort(key=lambda x: -x[0])
        total_reserved = len(reserved_tokens) + len(reserved_bytes)
        # We need to account for both winning tokens and winning templates
        # market_budget = max_vocab_size - total_reserved - len(bpe_tokens)
        
        # Split winners
        winning_tokens = []
        winning_templates_data = [] # List of word lists
        
        seen_tokens = set(reserved_tokens + reserved_bytes)
        
        # We iterate and pick until we hit max_vocab_size
        current_vocab_size = total_reserved + len(bpe_tokens)
        
        for freq, type, data in candidate_pool:
            if current_vocab_size >= max_vocab_size:
                break

            if type == 0: # Token
                # Strip leading/trailing non-space whitespace (\n, \r, \t, etc.)
                cleaned = data.strip(b'\t\n\x0b\x0c\r') if len(data) > 1 else data
                if len(cleaned) > 1 and cleaned not in seen_tokens:
                    winning_tokens.append(cleaned)
                    seen_tokens.add(cleaned)
                    current_vocab_size += 1
            elif type == 1: # Template
                winning_templates_data.append(data)
                current_vocab_size += 1

        # Add BPE tokens (ensuring they fit)
        for t in bpe_tokens:
            cleaned = t.strip(b'\t\n\x0b\x0c\r') if len(t) > 1 else t
            if len(cleaned) > 1 and cleaned not in seen_tokens:
                if len(winning_tokens) + len(reserved_tokens) + len(reserved_bytes) + len(winning_templates_data) < max_vocab_size:
                    winning_tokens.append(cleaned)
                    seen_tokens.add(cleaned)

        self.vocab = reserved_tokens + reserved_bytes + winning_tokens
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}

        # Map winning templates to final IDs
        self.templates = {}
        base_id = len(self.vocab)
        valid_template_count = 0
        
        for words in winning_templates_data:
            frame = []
            slot_positions = []
            valid = True
            for i, w in enumerate(words):
                if w == ';?':
                    frame.append(SLOT_SENTINEL)
                    slot_positions.append(i)
                else:
                    b = w.encode('latin-1')
                    if b in self.token_to_id:
                        frame.append(self.token_to_id[b])
                    else:
                        valid = False
                        break
            if valid:
                vid = base_id + valid_template_count
                self.templates[vid] = TemplateInfo(
                    vocab_id=vid,
                    length=len(frame),
                    num_slots=len(slot_positions),
                    frame=frame,
                    slot_positions=slot_positions
                )
                valid_template_count += 1

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

    def _build_template_index(self):
        """Group templates by first literal token ID for O(1) lookup.

        Each bucket is sorted by template length descending (longest-match-first).
        """
        self.template_index = {}
        for tmpl in self.templates.values():
            first_tok = tmpl.frame[0]  # slot position 0 excluded, so always literal
            if first_tok not in self.template_index:
                self.template_index[first_tok] = []
            self.template_index[first_tok].append(tmpl)
        # Sort each bucket: longest templates first
        for bucket in self.template_index.values():
            bucket.sort(key=lambda t: -t.length)

    def _apply_templates(self, tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Greedy left-to-right template matching on a token stream.

        Returns (new_tokens: uint16[], meta: uint8[]) where meta values are:
          0 = normal token, 1 = template ID, 2 = fill token.
        """
        n = len(tokens)
        out_tokens = []
        out_meta = []
        pos = 0
        matches = 0

        while pos < n:
            matched = False
            candidates = self.template_index.get(int(tokens[pos]), [])
            for tmpl in candidates:  # longest first
                L = tmpl.length
                if pos + L > n:
                    continue
                # Check all literal positions match
                ok = True
                for j in range(L):
                    if tmpl.frame[j] == SLOT_SENTINEL:
                        continue
                    if int(tokens[pos + j]) != tmpl.frame[j]:
                        ok = False
                        break
                if ok:
                    out_tokens.append(tmpl.vocab_id)
                    out_meta.append(1)  # template
                    for sp in tmpl.slot_positions:
                        out_tokens.append(int(tokens[pos + sp]))
                        out_meta.append(2)  # fill
                    pos += L
                    matched = True
                    matches += 1
                    break
            if not matched:
                out_tokens.append(int(tokens[pos]))
                out_meta.append(0)  # normal
                pos += 1

        print(f"  Templates matched {matches:,} times, "
              f"tokens {n:,} -> {len(out_tokens):,} "
              f"(saved {n - len(out_tokens):,})", flush=True)
        return (np.array(out_tokens, dtype=np.uint16),
                np.array(out_meta, dtype=np.uint8))

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
        If templates are loaded, applies template matching as a post-pass.
        Optionally writes a uint8 meta array (0=normal, 1=template, 2=fill).
        """
        # Event model preprocessing
        if self.event_model:
            event_file = input_path + "_event.txt"
            if not os.path.exists(event_file):
                print(f"Transforming input to event model text...", flush=True)
                text = open(input_path, 'r', encoding='utf-8', errors='replace').read()
                event_text = self._event_parser.transform(text)
                with open(event_file, 'w', encoding='utf-8') as f:
                    f.write(event_text)
            input_path = event_file

        print(f"Encoding {input_path} to {output_path} using fast_encoder...", flush=True)
        encoder_bin = self._ensure_fast_encoder()

        # Ensure vocab is saved for C to read
        if not os.path.exists(self.model_path):
            self.save(self.model_path)

        try:
            subprocess.run([encoder_bin, self.model_path, input_path, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"fast_encoder failed: {e}", flush=True)
            raise

        # Apply templates as a post-pass if we have any
        if self.templates:
            template_bin = self._ensure_fast_template()
            tmpl_path = self.model_path.replace('.vocab', '.templates') if '.vocab' in self.model_path else self.model_path + '.templates'
            if not os.path.exists(tmpl_path):
                self.save(self.model_path)
            cmd = [template_bin, tmpl_path, output_path, output_path]
            if output_meta_path:
                cmd.append(output_meta_path)
            subprocess.run(cmd, check=True)

    def decode(self, ids: List[int]) -> str:
        res = []
        i = 0
        while i < len(ids):
            tid = ids[i]
            if tid in self.templates:
                tmpl = self.templates[tid]
                # Consume next num_slots tokens as fills
                fills = ids[i + 1 : i + 1 + tmpl.num_slots]
                # Reconstruct by substituting fills into the frame
                for j, frame_tid in enumerate(tmpl.frame):
                    if frame_tid == SLOT_SENTINEL:
                        # Find which slot this is
                        slot_idx = tmpl.slot_positions.index(j)
                        fill_tid = fills[slot_idx] if slot_idx < len(fills) else 0
                        if 0 <= fill_tid < len(self.vocab):
                            res.append(self.vocab[fill_tid])
                    else:
                        if 0 <= frame_tid < len(self.vocab):
                            res.append(self.vocab[frame_tid])
                i += 1 + tmpl.num_slots
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

        # Save templates companion file
        if self.templates:
            tmpl_path = path.replace('.vocab', '.templates') if '.vocab' in path else path + '.templates'
            sorted_tmpls = sorted(self.templates.values(), key=lambda t: t.vocab_id)
            base_id = sorted_tmpls[0].vocab_id if sorted_tmpls else 0
            with open(tmpl_path, 'wb') as f:
                f.write(struct.pack('>I', len(sorted_tmpls)))
                f.write(struct.pack('>I', base_id))
                for tmpl in sorted_tmpls:
                    f.write(struct.pack('BB', tmpl.length, tmpl.num_slots))
                    for tid in tmpl.frame:
                        f.write(struct.pack('>H', tid))
            print(f"Saved {len(sorted_tmpls)} templates to {tmpl_path}", flush=True)
                
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

        # Load templates companion file if it exists
        tmpl_path = path.replace('.vocab', '.templates') if '.vocab' in path else path + '.templates'
        self.templates = {}
        self.template_index = {}
        if os.path.exists(tmpl_path):
            with open(tmpl_path, 'rb') as f:
                num_templates = struct.unpack('>I', f.read(4))[0]
                base_id = struct.unpack('>I', f.read(4))[0]
                for i in range(num_templates):
                    length, num_slots = struct.unpack('BB', f.read(2))
                    frame = [struct.unpack('>H', f.read(2))[0] for _ in range(length)]
                    slot_positions = [j for j, tid in enumerate(frame) if tid == SLOT_SENTINEL]
                    vid = base_id + i
                    self.templates[vid] = TemplateInfo(
                        vocab_id=vid,
                        length=length,
                        num_slots=num_slots,
                        frame=frame,
                        slot_positions=slot_positions,
                    )
            if self.templates:
                self._build_template_index()
            print(f"Loaded {len(self.templates)} templates from {tmpl_path}", flush=True)
            
    def _ensure_fast_template(self):
        """Compile fast_template if needed and return its path."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        template_bin = os.path.join(base_dir, "fast_template")
        if not os.path.exists(template_bin):
            print("Compiling fast_template...", flush=True)
            subprocess.run(
                ["gcc", "-O3", "-o", template_bin, os.path.join(base_dir, "fast_template.c")],
                check=True
            )
        return template_bin

if __name__ == "__main__":
    # Demo
    t = GreedyPhraseTokenizer(vocab_size=1000) # Small vocab for demo
    
    # Create a dummy training file
    with open("tokenizer/train_dummy.txt", "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 10)
        f.write("guest appearance " * 20)
        f.write("Hello world. " * 10)
    
    t.train(["tokenizer/train_dummy.txt"], bpe_slots=100)
    
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