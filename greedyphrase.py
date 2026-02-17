import os
import struct
import collections
import tempfile
import subprocess
from typing import List, Dict, Tuple


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

    def train(self, file_paths: List[str], phrase_ratio=0.95):
        """
        Trains the tokenizer vocabulary from a list of files.
        Uses C-based fast_counter for phrase extraction, then BPE on residuals.
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
                elif line == "BIGRAMS":
                    current_section = "phrases"
                    continue
                elif line == "TRIGRAMS":
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

        # ── Step A: Select phrases ──
        print("Selecting phrases for vocabulary...", flush=True)
        reserved_tokens = [b"<pad>", b"<unk>", b"<s>", b"</s>"]
        reserved_bytes = [bytes([i]) for i in range(256)]
        total_reserved = len(reserved_tokens) + len(reserved_bytes)  # 260
        remaining_slots = self.vocab_size - total_reserved  # 65276

        num_phrase_slots = int(remaining_slots * phrase_ratio)
        num_bpe_slots = remaining_slots - num_phrase_slots

        # Merge atom + phrase counts together — phrases (bigrams/trigrams) that are
        # multi-byte get priority; single-byte atoms are already covered by reserved_bytes
        all_phrase_freqs = collections.Counter()
        for atom, freq in atom_freqs.items():
            b = atom.encode('latin-1')
            if len(b) > 1:
                all_phrase_freqs[b] = freq
        for phrase, freq in phrase_freqs.items():
            b = phrase.encode('latin-1')
            all_phrase_freqs[b] += freq

        # Pick top phrases by frequency
        top_phrases = all_phrase_freqs.most_common(num_phrase_slots)
        phrase_entries = [b for b, _ in top_phrases]

        print(f"  Selected {len(phrase_entries)} phrases.", flush=True)

        # ── Step B: First-pass encode to collect residuals ──
        print("First-pass encode to collect residuals...", flush=True)
        partial_vocab = reserved_tokens + reserved_bytes + phrase_entries

        # Save partial vocab to temp file
        tmp_vocab_fd, tmp_vocab_path = tempfile.mkstemp(suffix='.vocab')
        os.close(tmp_vocab_fd)
        tmp_tokens_fd, tmp_tokens_path = tempfile.mkstemp(suffix='.tokens')
        os.close(tmp_tokens_fd)

        try:
            self._save_vocab(partial_vocab, tmp_vocab_path)

            # Run fast_encoder with partial vocab
            encoder_bin = self._ensure_fast_encoder()
            print(f"  Running fast_encoder (partial vocab, {len(partial_vocab)} tokens)...", flush=True)
            subprocess.run(
                [encoder_bin, tmp_vocab_path, target_file, tmp_tokens_path],
                check=True
            )

            # Read token stream and extract byte-fallback runs
            token_data = open(tmp_tokens_path, 'rb').read()
            num_tokens = len(token_data) // 2
            print(f"  First-pass produced {num_tokens:,} tokens.", flush=True)

            residuals = []
            current_run = bytearray()

            for offset in range(0, len(token_data), 2):
                tid = int.from_bytes(token_data[offset:offset+2], 'little')
                if 4 <= tid <= 259:
                    # Byte fallback token — accumulate
                    current_run.append(tid - 4)
                else:
                    if len(current_run) > 1:
                        residuals.append(bytes(current_run))
                    current_run = bytearray()

            # Don't forget trailing run
            if len(current_run) > 1:
                residuals.append(bytes(current_run))

            print(f"  Collected {len(residuals):,} residual segments "
                  f"({sum(len(r) for r in residuals):,} bytes total).", flush=True)

        finally:
            os.unlink(tmp_vocab_path)
            os.unlink(tmp_tokens_path)

        # ── Step C: Train BPE on residuals ──
        print(f"Training BPE on residuals ({num_bpe_slots} merges)...", flush=True)
        bpe_tokens = train_bpe(residuals, num_bpe_slots)

        # Filter out duplicates and tokens too long for the C encoder (MAX_TOKEN_LEN=1024)
        existing = set(reserved_tokens + reserved_bytes + phrase_entries)
        bpe_tokens = [t for t in bpe_tokens if t not in existing and len(t) < 1024]
        print(f"  Got {len(bpe_tokens)} unique BPE tokens.", flush=True)

        # ── Build final vocab ──
        self.vocab = reserved_tokens + reserved_bytes + phrase_entries + bpe_tokens

        # Pad or truncate to vocab_size
        if len(self.vocab) < self.vocab_size:
            # Pad with empty placeholder bytes (unused slots)
            pass  # It's fine to have fewer; encoder just won't use those IDs
        elif len(self.vocab) > self.vocab_size:
            self.vocab = self.vocab[:self.vocab_size]

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self._build_trie()

        print(f"Vocab size: {len(self.vocab)}", flush=True)
        print(f"  - Special: {len(reserved_tokens)}", flush=True)
        print(f"  - Bytes: {len(reserved_bytes)}", flush=True)
        print(f"  - Phrases: {len(phrase_entries)}", flush=True)
        print(f"  - BPE: {len(bpe_tokens)}", flush=True)
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

    def encode_file(self, input_path: str, output_path: str):
        """
        Encodes a large file using the C fast_encoder binary.
        """
        print(f"Encoding {input_path} to {output_path} using fast_encoder...", flush=True)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        encoder_bin = os.path.join(base_dir, "fast_encoder")
        
        if not os.path.exists(encoder_bin):
            print("Compiling fast_encoder...", flush=True)
            try:
                subprocess.run(
                    ["gcc", "-O3", "-o", encoder_bin, os.path.join(base_dir, "fast_encoder.c")],
                    check=True
                )
            except Exception as e:
                print(f"Failed to compile fast_encoder: {e}", flush=True)
                raise

        # Ensure vocab is saved for C to read
        if not os.path.exists(self.model_path):
            self.save(self.model_path)
            
        try:
            subprocess.run([encoder_bin, self.model_path, input_path, output_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"fast_encoder failed: {e}", flush=True)
            raise

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            if 0 <= i < len(self.vocab):
                res.append(self.vocab[i])
            else:
                res.append(b"") # Ignore invalid IDs
        return b"".join(res).decode('utf-8', errors='replace')

    def save(self, path):
        with open(path, 'wb') as f:
            # Simple serialization: newline separated bytes
            # Escape newlines in content for storage if needed, but for simplicity
            # we'll just pickle or use a length-prefixed format.
            # Let's use length-prefixed for safety.
            f.write(struct.pack('>I', len(self.vocab)))
            for token in self.vocab:
                f.write(struct.pack('>I', len(token)))
                f.write(token)
                
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

if __name__ == "__main__":
    # Demo
    t = GreedyPhraseTokenizer(vocab_size=1000) # Small vocab for demo
    
    # Create a dummy training file
    with open("tokenizer/train_dummy.txt", "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. " * 10)
        f.write("guest appearance " * 20)
        f.write("Hello world. " * 10)
    
    t.train(["tokenizer/train_dummy.txt"], phrase_ratio=0.5)
    
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
