import os
import struct
import collections
import re
import subprocess
from typing import List, Dict, Tuple

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
            
    def train(self, file_paths: List[str], phrase_ratio=0.5):
        """
        Trains the tokenizer vocabulary from a list of files.
        Uses C-based fast_counter for speed.
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

        # Run fast_counter on each file
        # For simplicity, we concatenate files or run sequentially.
        # Since fast_counter takes one file argument, let's run it on the first file (assuming one large file like tiny_stories)
        # Or loop if multiple.
        
        counts_file = os.path.join(base_dir, "tokenizer", "counts.txt")
        if os.path.exists(counts_file):
            os.remove(counts_file)
            
        # Run on the first file (assuming single training file for now)
        # If multiple, we'd need to merge counts. Let's support just the first file for the demo.
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
                    # Token is the rest of the line, unescaped
                    raw_token = " ".join(parts[1:])
                    token = raw_token.replace('\\\\', '\\').replace('\\n', '\n').replace('\\r', '\r')
                    
                    if current_section == "atoms":
                        atom_freqs[token] += count
                    elif current_section == "phrases":
                        phrase_freqs[token] += count
                except ValueError:
                    continue # Skip malformed lines if any
        
        print(f"Loaded {len(atom_freqs)} atoms and {len(phrase_freqs)} phrases.", flush=True)

        # 2. Vocabulary Selection
        print("Selecting best tokens for vocabulary...", flush=True)
        num_phrases = int(self.vocab_size * phrase_ratio)
        # Reserve space for 256 bytes + special tokens
        reserved_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        reserved_bytes = [bytes([i]) for i in range(256)]
        
        # Calculate remaining slots
        total_reserved = len(reserved_tokens) + len(reserved_bytes)
        remaining_slots = self.vocab_size - total_reserved
        
        # Split remaining slots between common atoms (words) and phrases
        # We'll give priority to phrases as requested, but we need atoms for coverage
        # Let's stick to the ratio for the *variable* part
        limit_phrases = int(remaining_slots * phrase_ratio)
        limit_atoms = remaining_slots - limit_phrases
        
        self.vocab = [t.encode('utf-8') for t in reserved_tokens] + reserved_bytes
        
        # Fill Atoms (Single words/punctuation)
        common_atoms = atom_freqs.most_common(limit_atoms)
        for atom, freq in common_atoms:
            # Avoid duplicates if atom is already a single byte
            b_atom = atom.encode('latin-1')
            if len(b_atom) > 1:
                self.vocab.append(b_atom)
            else:
                # Give the slot back to phrases if we skip an atom
                limit_phrases += 1
            
        # Fill Phrases (Multi-word strings)
        common_phrases = phrase_freqs.most_common(limit_phrases)
        for phrase, freq in common_phrases:
            self.vocab.append(phrase.encode('latin-1'))
            
        # Rebuild mappings
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self._build_trie()
        
        print(f"Vocab size: {len(self.vocab)}", flush=True)
        print(f"  - Bytes: 256", flush=True)
        print(f"  - Atoms: {len(common_atoms)}", flush=True)
        print(f"  - Phrases: {len(common_phrases)}", flush=True)
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
        print("✅ 'guest appearance' is a single token!")
    else:
        print("❌ 'guest appearance' was split.")
