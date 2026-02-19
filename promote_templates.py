import sys

def promote(input_path, output_path, budget=2000):
    templates = []
    with open(input_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue
            try:
                count = int(parts[0])
                tokens = parts[1:]
                num_slots = tokens.count(';?')
                length = len(tokens)
                # score = count * tokens_saved
                # tokens_saved = length - (1 + num_slots)
                # But wait, each slot is a token too.
                # If we replace L tokens with 1 template_id + S slot_tokens,
                # we save L - (1 + S) tokens.
                saved = length - (1 + num_slots)
                if saved <= 0:
                    continue
                score = count * saved
                templates.append((score, count, tokens))
            except ValueError:
                continue

    # Sort by score descending
    templates.sort(key=lambda x: -x[0])

    with open(output_path, 'w', encoding='latin-1') as f:
        for score, count, tokens in templates[:budget]:
            f.write(f"{count} {' '.join(tokens)}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 promote_templates.py <templates.txt> <winners.txt> [budget]")
    else:
        budget = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
        promote(sys.argv[1], sys.argv[2], budget)
