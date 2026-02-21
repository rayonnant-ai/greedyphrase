"""
Phase 12: 2D Tag-Quality Matrix Tokenizer

Converts raw text into a lossless coordinate stream of (Tag, Quality) pairs.
Tags represent semantic identity (P# entities, A# actions, O# objects, D# destinations).
Quality represents surface-form state (BASE, PAST, DEF, INDEF, etc.).

Uses spaCy for NLP (entity recognition, POS tagging, morphology, dependency parsing).
"""

import io
import json
import keyword
import re
import tokenize as py_tokenize
import spacy
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TagEntry:
    tag_id: str       # "P1", "A1", "O1", "D1"
    base_form: str    # "Ben", "walk", "vase", "store"
    category: str     # "Entity", "Action", "Object", "Destination"


@dataclass
class Coordinate:
    tag: str           # tag_id like "P1" or surface text like "through", "."
    quality: str       # closed 26-value axis: BASE, PAST, PREP, PUNCT, ADJ, WS, etc.
    surface: str = ""  # exact surface text for lossless reconstruction
    modifier: str = "" # stored modifier text (article) for lossless noun-phrase decode


# The 26 closed quality values
SEMANTIC_QUALITIES = {
    "BASE", "SUBJ", "OBJ", "POSS", "DEF", "INDEF",
    "PAST", "PRESENT", "PAST_CONT", "PRES_CONT", "PASSIVE_PAST", "PASSIVE_PRES",
}
FUNCTIONAL_QUALITIES = {
    "PREP", "CONJ", "PUNCT", "ADV", "DET", "AUX",
    "INTJ", "PRON", "ADJ", "PROPN", "NUM", "PART", "WORD", "WS",
}
CODE_QUALITIES = {"KW", "OP", "STR", "LITERAL", "COMMENT", "NL", "INDENT", "DEDENT"}
ALL_QUALITIES = SEMANTIC_QUALITIES | FUNCTIONAL_QUALITIES | CODE_QUALITIES

# --- 1D Vocab Layout (computed from ALL_QUALITIES) ---
VOCAB_SIZE = 16_384
N_QUALITIES = len(ALL_QUALITIES) + 1      # +1 for MOD
Q_MOD = N_QUALITIES - 1                   # modifier signal quality ID
BYTE_OFFSET = N_QUALITIES
LEXEME_OFFSET = BYTE_OFFSET + 256
MAX_LEXEMES = VOCAB_SIZE - LEXEME_OFFSET

# Stable alphabetical ordering of qualities
QUALITY_NAMES = sorted(ALL_QUALITIES)


class Vocab:
    """Maps (tag, quality) coordinates to/from a 1D integer token stream.

    Layout (16,384 IDs):
        0-26:       27 quality tokens (26 qualities + MOD)
        27-282:     256 byte tokens
        283-16383:  16,101 lexeme slots (most frequent tags)

    Unknown/overflow lexemes are byte-spelled using their UTF-8 bytes.
    """

    def __init__(self):
        self.quality_to_id = {}
        self.id_to_quality = {}
        self.lexeme_to_id = {}
        self.id_to_lexeme = {}

    @classmethod
    def build(cls, streams):
        """Build vocab from one or more coordinate streams.

        Assigns IDs to the 27 quality values and the top-16,101 most
        frequent tag/modifier strings.
        """
        vocab = cls()

        # Quality mapping: alphabetical order for the 26 qualities, MOD at 26
        for i, name in enumerate(QUALITY_NAMES):
            vocab.quality_to_id[name] = i
            vocab.id_to_quality[i] = name
        vocab.quality_to_id["MOD"] = Q_MOD
        vocab.id_to_quality[Q_MOD] = "MOD"

        # Count tag and modifier frequencies across all streams
        freq = defaultdict(int)
        for stream in streams:
            for coord in stream:
                freq[coord.tag] += 1
                if coord.modifier:
                    freq[coord.modifier] += 1

        # Assign lexeme IDs to the most frequent strings
        sorted_lexemes = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for i, (lexeme, _) in enumerate(sorted_lexemes[:MAX_LEXEMES]):
            token_id = LEXEME_OFFSET + i
            vocab.lexeme_to_id[lexeme] = token_id
            vocab.id_to_lexeme[token_id] = lexeme

        return vocab

    def _encode_text(self, text):
        """Encode text as a single lexeme ID or a sequence of UTF-8 byte IDs."""
        if text in self.lexeme_to_id:
            return [self.lexeme_to_id[text]]
        return [BYTE_OFFSET + b for b in text.encode("utf-8")]

    def encode_1d(self, stream):
        """Encode coordinate stream to 1D integer token list.

        Each coordinate emits tokens ending with a quality ID:
        - Known lexeme:   [lexeme_id, quality_id]
        - Unknown lexeme:  [byte0, ..., byteN, quality_id]
        - With modifier:   [mod_tokens, Q_MOD, tag_tokens, quality_id]
        """
        ids = []
        for coord in stream:
            if coord.modifier:
                ids.extend(self._encode_text(coord.modifier))
                ids.append(Q_MOD)
            ids.extend(self._encode_text(coord.tag))
            ids.append(self.quality_to_id[coord.quality])
        return ids

    def decode_1d(self, ids):
        """Decode 1D integer token list back to Coordinate objects.

        Accumulates byte/lexeme tokens; a quality token terminates each
        coordinate. Q_MOD signals a modifier prefix for the next coordinate.
        """
        result = []
        pending_bytes = []
        pending_lexeme = None
        pending_modifier = ""

        for token_id in ids:
            if token_id < N_QUALITIES:
                # Quality token — resolve accumulated tag
                if pending_lexeme is not None:
                    tag = pending_lexeme
                elif pending_bytes:
                    tag = bytes(pending_bytes).decode("utf-8")
                else:
                    tag = ""

                quality_name = self.id_to_quality[token_id]

                if quality_name == "MOD":
                    pending_modifier = tag
                else:
                    result.append(Coordinate(
                        tag=tag,
                        quality=quality_name,
                        modifier=pending_modifier,
                    ))
                    pending_modifier = ""

                pending_bytes = []
                pending_lexeme = None

            elif token_id < LEXEME_OFFSET:
                # Byte token
                pending_bytes.append(token_id - BYTE_OFFSET)

            else:
                # Lexeme token
                pending_lexeme = self.id_to_lexeme[token_id]

        return result

    def save(self, path):
        """Persist vocab to a JSON file."""
        data = {
            "qualities": self.quality_to_id,
            "lexemes": {str(v): k for k, v in self.lexeme_to_id.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load vocab from a JSON file."""
        vocab = cls()
        with open(path, "r") as f:
            data = json.load(f)
        for name, qid in data["qualities"].items():
            vocab.quality_to_id[name] = qid
            vocab.id_to_quality[qid] = name
        for lid_str, lexeme in data["lexemes"].items():
            lid = int(lid_str)
            vocab.lexeme_to_id[lexeme] = lid
            vocab.id_to_lexeme[lid] = lexeme
        return vocab

    def stats(self, stream, ids):
        """Return encoding statistics."""
        n_coords = len(stream)
        n_tokens = len(ids)
        n_byte_spelled = sum(
            1 for coord in stream
            if coord.tag not in self.lexeme_to_id
            and coord.tag != ""
        )
        n_mod_byte_spelled = sum(
            1 for coord in stream
            if coord.modifier
            and coord.modifier not in self.lexeme_to_id
        )
        return {
            "coordinates": n_coords,
            "tokens_1d": n_tokens,
            "tokens_per_coord": n_tokens / n_coords if n_coords else 0,
            "lexemes_in_vocab": len(self.lexeme_to_id),
            "byte_spelled_tags": n_byte_spelled,
            "byte_spelled_modifiers": n_mod_byte_spelled,
            "byte_spell_rate": (n_byte_spelled + n_mod_byte_spelled) / n_coords if n_coords else 0,
        }


# Pronoun mappings
SUBJECT_PRONOUNS = {"he", "she", "it", "they"}
OBJECT_PRONOUNS = {"him", "her", "it", "them"}
POSSESSIVE_PRONOUNS = {"his", "her", "its", "their"}
POSSESSIVE_DETERMINERS = {"his", "her", "its", "their"}

# Gender mapping for pronoun resolution
MALE_PRONOUNS = {"he", "him", "his"}
FEMALE_PRONOUNS = {"she", "her", "hers"}
NEUTRAL_PRONOUNS = {"it", "its"}

# Destination indicators (spaCy dep labels or preposition context)
DESTINATION_PREPS = {"to", "into", "through", "toward", "towards"}
LOCATION_LABELS = {"GPE", "LOC", "FAC"}


class TagQualityTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def encode(self, text: str) -> tuple[dict, list]:
        """Returns (tag_dictionary, coordinate_stream)"""
        doc = self.nlp(text)
        tag_dict = {}
        stream = []

        # Counters for tag ID generation
        counters = {"P": 0, "A": 0, "O": 0, "D": 0}

        # Mappings: lemma/name → tag_id
        entity_map = {}   # entity text → tag_id
        action_map = {}   # verb lemma → tag_id
        object_map = {}   # noun lemma → tag_id
        dest_map = {}     # destination lemma → tag_id

        # Pronoun resolution tracking
        last_male_entity = None
        last_female_entity = None
        last_neutral_entity = None

        def make_tag(prefix, base_form, category):
            counters[prefix] += 1
            tag_id = f"{prefix}{counters[prefix]}"
            tag_dict[tag_id] = TagEntry(tag_id=tag_id, base_form=base_form, category=category)
            return tag_id

        # --- Pass 1: Entity Discovery ---

        # Named entities from spaCy NER → P# (persons), D# (locations)
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if not ent_text:
                continue
            if ent.label_ == "PERSON":
                if ent_text not in entity_map:
                    tag_id = make_tag("P", ent_text, "Entity")
                    entity_map[ent_text] = tag_id
            elif ent.label_ in LOCATION_LABELS:
                if ent_text not in dest_map:
                    tag_id = make_tag("D", ent_text, "Destination")
                    dest_map[ent_text] = tag_id

        # Scan tokens for verbs → A# tags, nouns → O#/D# tags
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ != "aux":
                lemma = token.lemma_.lower()
                if lemma not in action_map:
                    tag_id = make_tag("A", lemma, "Action")
                    action_map[lemma] = tag_id

        # Also check for past-participle-like adjectives after "be" (e.g. "was amazed")
        for token in doc:
            if (token.pos_ == "ADJ" and token.dep_ in ("acomp", "attr")
                    and token.head.lemma_ == "be"
                    and (token.text.endswith("ed") or token.text.endswith("en"))):
                # Derive verb lemma: "amazed" → "amaze", "walked" → "walk"
                verb_lemma = self._derive_verb_lemma(token.text)
                if verb_lemma not in action_map:
                    tag_id = make_tag("A", verb_lemma, "Action")
                    action_map[verb_lemma] = tag_id

        # Nouns: classify as D# if preceded by destination prep, else O#
        # First collect which nouns appear as destinations
        dest_nouns = set()
        for token in doc:
            if token.pos_ == "NOUN":
                # Check if this noun's head is a destination prep
                if token.head.pos_ == "ADP" and token.head.text.lower() in DESTINATION_PREPS:
                    dest_nouns.add(token.lemma_.lower())
                # Also check direct prep children
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() in DESTINATION_PREPS:
                        dest_nouns.add(token.lemma_.lower())

        for token in doc:
            if token.pos_ == "NOUN":
                lemma = token.lemma_.lower()
                # Skip if already captured as named entity destination
                if lemma in dest_map or any(lemma in e.lower() for e in entity_map):
                    continue
                if lemma in dest_nouns:
                    if lemma not in dest_map:
                        # Use the token's text for base form (capitalized if original was)
                        tag_id = make_tag("D", lemma, "Destination")
                        dest_map[lemma] = tag_id
                else:
                    if lemma not in object_map:
                        tag_id = make_tag("O", lemma, "Object")
                        object_map[lemma] = tag_id

        # --- Pass 2: Build coordinate stream ---

        tokens = list(doc)
        i = 0
        prev_end = 0  # track position in original text for whitespace

        while i < len(tokens):
            token = tokens[i]

            # Handle whitespace before this token
            ws_before = text[prev_end:token.idx]
            if ws_before and ws_before != " ":
                # Non-standard whitespace — emit explicit WS
                stream.append(Coordinate(tag=ws_before, quality="WS", surface=ws_before))
            elif ws_before == " " and stream:
                # Standard single space — implicit, no emission needed
                pass
            elif ws_before == "" and stream:
                # No space between tokens — emit empty WS to suppress default space
                stream.append(Coordinate(tag="", quality="WS", surface=""))

            # --- Try to match multi-token patterns first ---

            # Check for "was/is + VERB-ing" → PAST_CONT/PRES_CONT
            if (token.pos_ == "AUX" and token.lemma_.lower() in ("be", "have")
                    and i + 1 < len(tokens)):
                next_tok = tokens[i + 1]
                # "was walking" → PAST_CONT
                if (next_tok.pos_ == "VERB" and next_tok.tag_ == "VBG"
                        and token.lemma_.lower() == "be"):
                    verb_lemma = next_tok.lemma_.lower()
                    if verb_lemma in action_map:
                        tag_id = action_map[verb_lemma]
                        morph = token.morph
                        if "Past" in morph.get("Tense", []):
                            quality = "PAST_CONT"
                        else:
                            quality = "PRES_CONT"
                        surface = token.text + text[token.idx + len(token.text):next_tok.idx] + next_tok.text
                        stream.append(Coordinate(tag=tag_id, quality=quality, surface=surface))
                        prev_end = next_tok.idx + len(next_tok.text)
                        i += 2
                        # Update entity tracking for pronoun resolution
                        continue

                # "was amazed" → PASSIVE_PAST (handles VBN or ADJ past-participle-like)
                if (token.lemma_.lower() == "be"
                        and next_tok.pos_ in ("VERB", "ADJ")
                        and (next_tok.tag_ == "VBN"
                             or (next_tok.pos_ == "ADJ"
                                 and (next_tok.text.endswith("ed") or next_tok.text.endswith("en"))))):
                    # Try direct lemma lookup, then derived verb lemma
                    verb_lemma = next_tok.lemma_.lower()
                    if verb_lemma not in action_map:
                        verb_lemma = self._derive_verb_lemma(next_tok.text)
                    if verb_lemma in action_map:
                        tag_id = action_map[verb_lemma]
                        surface = token.text + text[token.idx + len(token.text):next_tok.idx] + next_tok.text
                        morph = token.morph
                        if "Past" in morph.get("Tense", []):
                            quality = "PASSIVE_PAST"
                        else:
                            quality = "PASSIVE_PRES"
                        stream.append(Coordinate(tag=tag_id, quality=quality, surface=surface))
                        prev_end = next_tok.idx + len(next_tok.text)
                        i += 2
                        continue

            # Check for determiner + [adj/adv]* + noun → emit modifiers separately, noun with DEF/INDEF
            if token.pos_ == "DET" and token.text.lower() in ("the", "a", "an"):
                det_text = token.text
                det_lower = det_text.lower()
                # Look ahead for adjectives/adverbs and noun
                j = i + 1
                modifiers = []
                while j < len(tokens):
                    t = tokens[j]
                    if t.pos_ in ("ADJ", "ADV"):
                        modifiers.append(t)
                        j += 1
                    else:
                        break

                if j < len(tokens) and tokens[j].pos_ == "NOUN":
                    noun_tok = tokens[j]
                    noun_lemma = noun_tok.lemma_.lower()

                    # Find the tag for this noun
                    noun_tag = None
                    if noun_lemma in object_map:
                        noun_tag = object_map[noun_lemma]
                    elif noun_lemma in dest_map:
                        noun_tag = dest_map[noun_lemma]

                    if noun_tag is not None:
                        # Advance prev_end past the absorbed article
                        prev_end = token.idx + len(token.text)

                        if det_lower == "the":
                            quality = "DEF"
                        else:
                            quality = "INDEF"

                        if modifiers:
                            # Store article on the FIRST modifier for lossless decode
                            for mi, mod_tok in enumerate(modifiers):
                                ws_before_mod = text[prev_end:mod_tok.idx]
                                if ws_before_mod and ws_before_mod != " ":
                                    stream.append(Coordinate(tag=ws_before_mod, quality="WS", surface=ws_before_mod))
                                elif ws_before_mod == "" and stream:
                                    stream.append(Coordinate(tag="", quality="WS", surface=""))

                                q = "ADV" if mod_tok.pos_ == "ADV" else "ADJ"
                                # First modifier carries the article text
                                mod_field = det_text if mi == 0 else ""
                                stream.append(Coordinate(
                                    tag=mod_tok.text, quality=q,
                                    surface=mod_tok.text, modifier=mod_field
                                ))
                                prev_end = mod_tok.idx + len(mod_tok.text)

                            # Handle whitespace before noun
                            ws_before_noun = text[prev_end:noun_tok.idx]
                            if ws_before_noun and ws_before_noun != " ":
                                stream.append(Coordinate(tag=ws_before_noun, quality="WS", surface=ws_before_noun))
                            elif ws_before_noun == "" and stream:
                                stream.append(Coordinate(tag="", quality="WS", surface=""))

                            # Noun with DEF/INDEF — no modifier field needed
                            stream.append(Coordinate(
                                tag=noun_tag, quality=quality,
                                surface=noun_tok.text
                            ))
                        else:
                            # No modifiers — store article on the noun
                            ws_before_noun = text[prev_end:noun_tok.idx]
                            if ws_before_noun and ws_before_noun != " ":
                                stream.append(Coordinate(tag=ws_before_noun, quality="WS", surface=ws_before_noun))
                            elif ws_before_noun == "" and stream:
                                stream.append(Coordinate(tag="", quality="WS", surface=""))

                            stream.append(Coordinate(
                                tag=noun_tag, quality=quality,
                                surface=noun_tok.text, modifier=det_text
                            ))

                        prev_end = noun_tok.idx + len(noun_tok.text)
                        i = j + 1

                        # Update neutral entity for "it" pronoun resolution
                        last_neutral_entity = noun_tag

                        continue

            # Named entity match (single token or multi-token)
            matched_entity = False
            for ent_text, tag_id in entity_map.items():
                if text[token.idx:token.idx + len(ent_text)] == ent_text:
                    # Count how many tokens this entity spans
                    ent_end = token.idx + len(ent_text)
                    span_end = i
                    pos = token.idx
                    while span_end < len(tokens) and tokens[span_end].idx < ent_end:
                        span_end += 1

                    surface = text[token.idx:ent_end]
                    stream.append(Coordinate(tag=tag_id, quality="BASE", surface=surface))

                    # Update pronoun resolution tracking
                    entry = tag_dict[tag_id]
                    if entry.category == "Entity":
                        # Simplified gender tracking — assume male for now
                        last_male_entity = tag_id

                    prev_end = ent_end
                    i = span_end
                    matched_entity = True
                    break

            if matched_entity:
                continue

            # Pronoun handling
            tok_lower = token.text.lower()
            if tok_lower in SUBJECT_PRONOUNS | OBJECT_PRONOUNS | POSSESSIVE_PRONOUNS:
                resolved = self._resolve_pronoun(
                    tok_lower, last_male_entity, last_female_entity, last_neutral_entity
                )
                if resolved:
                    # Use spaCy dep to distinguish subj vs obj for ambiguous pronouns like "it"
                    if tok_lower in POSSESSIVE_DETERMINERS:
                        quality = "POSS"
                    elif token.dep_ in ("nsubj", "nsubjpass"):
                        quality = "SUBJ"
                    elif token.dep_ in ("dobj", "pobj", "iobj"):
                        quality = "OBJ"
                    elif tok_lower in SUBJECT_PRONOUNS:
                        quality = "SUBJ"
                    else:
                        quality = "OBJ"
                    stream.append(Coordinate(tag=resolved, quality=quality, surface=token.text))
                    prev_end = token.idx + len(token.text)
                    i += 1
                    continue

            # Verb handling
            if token.pos_ == "VERB" and token.dep_ != "aux":
                verb_lemma = token.lemma_.lower()
                if verb_lemma in action_map:
                    tag_id = action_map[verb_lemma]
                    quality = self._verb_quality(token)
                    stream.append(Coordinate(tag=tag_id, quality=quality, surface=token.text))
                    prev_end = token.idx + len(token.text)
                    i += 1
                    continue

            # Bare noun handling (without determiner)
            if token.pos_ == "NOUN":
                noun_lemma = token.lemma_.lower()
                noun_tag = None
                if noun_lemma in object_map:
                    noun_tag = object_map[noun_lemma]
                elif noun_lemma in dest_map:
                    noun_tag = dest_map[noun_lemma]
                if noun_tag:
                    stream.append(Coordinate(tag=noun_tag, quality="BASE", surface=token.text))
                    last_neutral_entity = noun_tag
                    prev_end = token.idx + len(token.text)
                    i += 1
                    continue

            # --- Functional tokens (tag=surface, quality=CATEGORY) ---
            POS_TO_QUALITY = {
                "ADP": "PREP", "CCONJ": "CONJ", "SCONJ": "CONJ",
                "PUNCT": "PUNCT", "ADV": "ADV", "DET": "DET", "AUX": "AUX",
                "INTJ": "INTJ", "PRON": "PRON", "ADJ": "ADJ", "PROPN": "PROPN",
                "NUM": "NUM", "PART": "PART",
            }
            q = POS_TO_QUALITY.get(token.pos_, "WORD")
            stream.append(Coordinate(tag=token.text, quality=q, surface=token.text))

            prev_end = token.idx + len(token.text)
            i += 1

        # Handle trailing text after last token
        if prev_end < len(text):
            trailing = text[prev_end:]
            if trailing.strip():
                stream.append(Coordinate(tag=trailing, quality="WORD", surface=trailing))
            elif trailing:
                stream.append(Coordinate(tag=trailing, quality="WS", surface=trailing))

        return tag_dict, stream

    def decode(self, tag_dict: dict, stream: list) -> str:
        """Reconstructs text from coordinates — must be lossless.

        Article placement: when a coordinate has a modifier field, the stored
        text (article) is prepended before the surface with a space.
        - On ADJ/ADV: article before modifier (first in noun phrase)
        - On noun (DEF/INDEF): article before noun (bare det+noun)
        """
        parts = []
        for idx, coord in enumerate(stream):
            if coord.quality == "WS":
                parts.append(coord.surface)
                continue

            # Implicit space between non-WS tokens
            if parts:
                prev_is_ws = idx > 0 and stream[idx - 1].quality == "WS"
                if not prev_is_ws:
                    parts.append(" ")

            # Prepend stored article if present
            if coord.modifier:
                parts.append(coord.modifier)
                parts.append(" ")

            parts.append(coord.surface)

        return "".join(parts)

    def format_dictionary(self, tag_dict: dict) -> str:
        """Pretty-print the 2D dictionary as a table."""
        lines = []
        lines.append(f"{'Tag ID':<8} {'BASE Form':<20} {'Category':<15}")
        lines.append("-" * 45)
        for tag_id, entry in sorted(tag_dict.items(), key=lambda x: (x[1].category, x[0])):
            lines.append(f"{entry.tag_id:<8} {entry.base_form:<20} {entry.category:<15}")
        return "\n".join(lines)

    def format_stream(self, stream: list) -> str:
        """Pretty-print coordinate stream."""
        parts = []
        for coord in stream:
            if coord.quality == "WS":
                if coord.tag == " ":
                    parts.append("(WS)")
                elif coord.tag == "":
                    parts.append("(WS, \"\")")
                else:
                    parts.append(f"(WS, {repr(coord.tag)})")
            elif coord.quality in FUNCTIONAL_QUALITIES:
                parts.append(f"({repr(coord.tag)}, {coord.quality})")
            else:
                parts.append(f"({coord.tag}, {coord.quality})")
        return " ".join(parts)

    @staticmethod
    def _derive_verb_lemma(word: str) -> str:
        """Derive verb lemma from a past-participle-like form.

        Uses vowel/consonant pattern: if char before 'ed' has a vowel before it,
        the base form likely ends in 'e' → strip just 'd'. Otherwise strip 'ed'.
        E.g. "amazed" → "amaze", "walked" → "walk", "placed" → "place".
        """
        word = word.lower()
        vowels = set("aeiou")
        if word.endswith("ed") and len(word) > 4:
            # If 4th-from-end is vowel and 3rd-from-end is consonant: base ends in 'e'
            if word[-4] in vowels and word[-3] not in vowels:
                return word[:-1]   # "amazed" → "amaze", "placed" → "place"
            else:
                return word[:-2]   # "walked" → "walk", "blessed" → "bless"
        elif word.endswith("ed") and len(word) > 2:
            return word[:-2]
        elif word.endswith("en"):
            return word
        return word

    def _verb_quality(self, token) -> str:
        """Determine verb quality from spaCy morphology."""
        morph = token.morph
        tense = morph.get("Tense", [])
        aspect = morph.get("Aspect", [])
        verbform = morph.get("VerbForm", [])

        if "Past" in tense:
            return "PAST"
        if "Pres" in tense:
            return "PRESENT"
        if "Ger" in verbform:
            return "GERUND"
        if "Inf" in verbform:
            return "BASE"
        if "Part" in verbform and "Past" in tense:
            return "PAST_PART"
        return "BASE"

    def _resolve_pronoun(self, pronoun, last_male, last_female, last_neutral):
        """Resolve a pronoun to its antecedent entity tag."""
        pronoun = pronoun.lower()
        if pronoun in MALE_PRONOUNS:
            return last_male
        if pronoun in FEMALE_PRONOUNS:
            return last_female
        if pronoun in NEUTRAL_PRONOUNS:
            return last_neutral
        return None



# --- Keyword sets for code tokenizers ---
PYTHON_KEYWORDS = set(keyword.kwlist)

BASH_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done",
    "case", "esac", "in", "function", "select", "time", "coproc",
    "return", "exit", "break", "continue", "shift", "export", "local",
    "readonly", "declare", "typeset", "unset", "source", "eval", "exec",
    "trap", "set", "shopt",
}

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while", "_Bool", "_Complex", "_Imaginary",
}

# --- Regex patterns for Bash lexer ---
_BASH_TOKEN_RE = re.compile(r"""
    (?P<COMMENT>\#[^\n]*)                           |
    (?P<STRING_DQ>"(?:[^"\\]|\\.)*")                |
    (?P<STRING_SQ>'[^']*')                          |
    (?P<VAR_BRACE>\$\{[^}]*\})                      |
    (?P<VAR_SIMPLE>\$[A-Za-z_][A-Za-z0-9_]*)        |
    (?P<VAR_SPECIAL>\$[0-9@#?!\$\-*])               |
    (?P<NUMBER>\b[0-9]+(?:\.[0-9]+)?\b)             |
    (?P<SHEBANG>\#![^\n]*)                          |
    (?P<OPERATOR>[|&;(){}<>!]+|&&|\|\||;;|<<|>>|[=])  |
    (?P<KEYWORD>\b(?:""" + "|".join(re.escape(k) for k in sorted(BASH_KEYWORDS, key=lambda x: -len(x))) + r""")\b) |
    (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)                |
    (?P<NEWLINE>\n)                                  |
    (?P<WS>[ \t]+)                                   |
    (?P<OTHER>.)
""", re.VERBOSE)

# --- Regex patterns for C lexer ---
_C_TOKEN_RE = re.compile(r"""
    (?P<BLOCK_COMMENT>/\*[\s\S]*?\*/)                |
    (?P<LINE_COMMENT>//[^\n]*)                       |
    (?P<INCLUDE>\#\s*include\s*[<"][^>"]*[>"])        |
    (?P<PREPROC>\#\s*[a-z]+[^\n]*)                   |
    (?P<STRING>"(?:[^"\\]|\\.)*")                    |
    (?P<CHAR_LIT>'(?:[^'\\]|\\.)*')                  |
    (?P<NUMBER>\b[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?[fFlLuU]?\b|0[xX][0-9a-fA-F]+[uUlL]*\b) |
    (?P<OPERATOR>[+\-*/%=!<>&|^~?:;,.\[\]{}()])     |
    (?P<KEYWORD>\b(?:""" + "|".join(re.escape(k) for k in sorted(C_KEYWORDS, key=lambda x: -len(x))) + r""")\b) |
    (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)                |
    (?P<NEWLINE>\n)                                  |
    (?P<WS>[ \t]+)                                   |
    (?P<OTHER>.)
""", re.VERBOSE)


class CodeTokenizer:
    """Tokenizes Python, Bash, and C source code into Coordinate streams.

    Entity types: V# (variables), F# (functions), T# (types).
    All whitespace is explicit (WS/NL coords) — no implicit spaces.
    """

    def encode(self, code: str, lang: str) -> tuple[dict, list[Coordinate]]:
        lang = lang.lower()
        if lang == "python":
            return self._encode_python(code)
        elif lang == "bash":
            return self._encode_bash(code)
        elif lang == "c":
            return self._encode_c(code)
        else:
            raise ValueError(f"Unsupported language: {lang}")

    def decode(self, tag_dict: dict, stream: list[Coordinate]) -> str:
        return "".join(c.surface for c in stream)

    # --- Python encoder ---

    def _encode_python(self, code: str) -> tuple[dict, list[Coordinate]]:
        tag_dict = {}
        counters = {"V": 0, "F": 0, "T": 0}
        entity_map = {}  # name → tag_id

        def make_tag(prefix, base_form, category):
            counters[prefix] += 1
            tag_id = f"{prefix}{counters[prefix]}"
            tag_dict[tag_id] = TagEntry(tag_id=tag_id, base_form=base_form, category=category)
            return tag_id

        # Tokenize with stdlib
        tokens = list(py_tokenize.generate_tokens(io.StringIO(code).readline))

        # Pass 1: Entity discovery
        for idx, tok in enumerate(tokens):
            if tok.type == py_tokenize.NAME:
                name = tok.string
                if name in entity_map or keyword.iskeyword(name) or name in ("True", "False", "None"):
                    continue
                # def NAME → F#
                if idx > 0 and tokens[idx - 1].type == py_tokenize.NAME and tokens[idx - 1].string == "def":
                    entity_map[name] = make_tag("F", name, "Function")
                # class NAME → T#
                elif idx > 0 and tokens[idx - 1].type == py_tokenize.NAME and tokens[idx - 1].string == "class":
                    entity_map[name] = make_tag("T", name, "Type")
                # NAME( → F# (call)
                elif idx + 1 < len(tokens) and tokens[idx + 1].type == py_tokenize.OP and tokens[idx + 1].string == "(":
                    entity_map[name] = make_tag("F", name, "Function")
                # NAME = or for NAME in → V#
                elif idx + 1 < len(tokens) and tokens[idx + 1].type == py_tokenize.OP and tokens[idx + 1].string == "=":
                    entity_map[name] = make_tag("V", name, "Variable")
                elif (idx > 0 and tokens[idx - 1].type == py_tokenize.NAME
                      and tokens[idx - 1].string == "for"
                      and idx + 1 < len(tokens) and tokens[idx + 1].type == py_tokenize.NAME
                      and tokens[idx + 1].string == "in"):
                    entity_map[name] = make_tag("V", name, "Variable")

        # Build line-offset table for (line, col) → char offset conversion
        line_offsets = [0]  # line 1 starts at offset 0
        for i, ch in enumerate(code):
            if ch == "\n":
                line_offsets.append(i + 1)
        # Sentinel for tokens referencing one past the last line (e.g. DEDENT/ENDMARKER)
        if not code.endswith("\n"):
            line_offsets.append(len(code))

        def pos_to_offset(line, col):
            return line_offsets[line - 1] + col

        # Pass 2: Build coordinate stream
        stream = []
        prev_off = 0

        for tok in tokens:
            if tok.type == py_tokenize.ENCODING:
                continue
            if tok.type == py_tokenize.ENDMARKER:
                break

            text = tok.string
            start_off = pos_to_offset(*tok.start)
            end_off = pos_to_offset(*tok.end)

            # Emit gap between previous token end and this token start
            if start_off > prev_off:
                gap = code[prev_off:start_off]
                stream.append(Coordinate(tag=gap, quality="WS", surface=gap))

            # Map token to coordinate
            if tok.type == py_tokenize.INDENT:
                stream.append(Coordinate(tag=text, quality="INDENT", surface=text))
            elif tok.type == py_tokenize.DEDENT:
                stream.append(Coordinate(tag="", quality="DEDENT", surface=""))
            elif tok.type in (py_tokenize.NEWLINE, py_tokenize.NL):
                stream.append(Coordinate(tag=text, quality="NL", surface=text))
            elif tok.type == py_tokenize.COMMENT:
                stream.append(Coordinate(tag=text, quality="COMMENT", surface=text))
            elif tok.type == py_tokenize.STRING:
                stream.append(Coordinate(tag=text, quality="STR", surface=text))
            elif tok.type == py_tokenize.NUMBER:
                stream.append(Coordinate(tag=text, quality="LITERAL", surface=text))
            elif tok.type == py_tokenize.OP:
                stream.append(Coordinate(tag=text, quality="OP", surface=text))
            elif tok.type == py_tokenize.NAME:
                if keyword.iskeyword(text) or text in ("True", "False", "None"):
                    stream.append(Coordinate(tag=text, quality="KW", surface=text))
                elif text in entity_map:
                    stream.append(Coordinate(tag=entity_map[text], quality="BASE", surface=text))
                else:
                    stream.append(Coordinate(tag=text, quality="KW", surface=text))
            else:
                stream.append(Coordinate(tag=text, quality="WORD", surface=text))

            if end_off > prev_off:
                prev_off = end_off

        # Handle trailing content after last token
        if prev_off < len(code):
            trailing = code[prev_off:]
            stream.append(Coordinate(tag=trailing, quality="WS", surface=trailing))

        return tag_dict, stream

    # --- Bash encoder ---

    def _encode_bash(self, code: str) -> tuple[dict, list[Coordinate]]:
        tag_dict = {}
        counters = {"V": 0, "F": 0, "T": 0}
        entity_map = {}

        def make_tag(prefix, base_form, category):
            counters[prefix] += 1
            tag_id = f"{prefix}{counters[prefix]}"
            tag_dict[tag_id] = TagEntry(tag_id=tag_id, base_form=base_form, category=category)
            return tag_id

        raw_tokens = list(_BASH_TOKEN_RE.finditer(code))

        # Pass 1: Entity discovery
        for idx, m in enumerate(raw_tokens):
            kind = m.lastgroup
            text = m.group()
            if kind == "NAME":
                if text in entity_map or text in BASH_KEYWORDS:
                    continue
                # NAME= → V#
                if idx + 1 < len(raw_tokens) and raw_tokens[idx + 1].group() == "=":
                    entity_map[text] = make_tag("V", text, "Variable")
                # function NAME or NAME() → F#
                elif (idx > 0 and raw_tokens[idx - 1].lastgroup == "KEYWORD"
                      and raw_tokens[idx - 1].group() == "function"):
                    entity_map[text] = make_tag("F", text, "Function")
                elif (idx + 1 < len(raw_tokens)
                      and raw_tokens[idx + 1].lastgroup == "OPERATOR"
                      and raw_tokens[idx + 1].group() == "("):
                    entity_map[text] = make_tag("F", text, "Function")

        # Pass 2: Build stream
        stream = []
        for m in raw_tokens:
            kind = m.lastgroup
            text = m.group()

            if kind == "NEWLINE":
                stream.append(Coordinate(tag=text, quality="NL", surface=text))
            elif kind == "WS":
                stream.append(Coordinate(tag=text, quality="WS", surface=text))
            elif kind in ("COMMENT", "SHEBANG"):
                stream.append(Coordinate(tag=text, quality="COMMENT", surface=text))
            elif kind in ("STRING_DQ", "STRING_SQ"):
                stream.append(Coordinate(tag=text, quality="STR", surface=text))
            elif kind in ("VAR_BRACE", "VAR_SIMPLE", "VAR_SPECIAL"):
                # Variables referenced with $ — emit as surface
                stream.append(Coordinate(tag=text, quality="OP", surface=text))
            elif kind == "NUMBER":
                stream.append(Coordinate(tag=text, quality="LITERAL", surface=text))
            elif kind == "OPERATOR":
                stream.append(Coordinate(tag=text, quality="OP", surface=text))
            elif kind == "KEYWORD":
                stream.append(Coordinate(tag=text, quality="KW", surface=text))
            elif kind == "NAME":
                if text in entity_map:
                    stream.append(Coordinate(tag=entity_map[text], quality="BASE", surface=text))
                else:
                    stream.append(Coordinate(tag=text, quality="KW", surface=text))
            elif kind == "OTHER":
                stream.append(Coordinate(tag=text, quality="OP", surface=text))

        return tag_dict, stream

    # --- C encoder ---

    def _encode_c(self, code: str) -> tuple[dict, list[Coordinate]]:
        tag_dict = {}
        counters = {"V": 0, "F": 0, "T": 0}
        entity_map = {}

        def make_tag(prefix, base_form, category):
            counters[prefix] += 1
            tag_id = f"{prefix}{counters[prefix]}"
            tag_dict[tag_id] = TagEntry(tag_id=tag_id, base_form=base_form, category=category)
            return tag_id

        raw_tokens = list(_C_TOKEN_RE.finditer(code))

        # Pass 1: Entity discovery
        for idx, m in enumerate(raw_tokens):
            kind = m.lastgroup
            text = m.group()
            if kind == "NAME":
                if text in entity_map or text in C_KEYWORDS:
                    continue
                # struct NAME → T#
                if (idx > 0 and raw_tokens[idx - 1].lastgroup == "KEYWORD"
                        and raw_tokens[idx - 1].group() == "struct"):
                    entity_map[text] = make_tag("T", text, "Type")
                # TYPE NAME( → F#
                elif (idx + 1 < len(raw_tokens)
                      and raw_tokens[idx + 1].lastgroup == "OPERATOR"
                      and raw_tokens[idx + 1].group() == "("):
                    # Check if preceded by a type keyword or name (skip WS)
                    prev_idx = idx - 1
                    while prev_idx >= 0 and raw_tokens[prev_idx].lastgroup == "WS":
                        prev_idx -= 1
                    if prev_idx >= 0 and raw_tokens[prev_idx].lastgroup in ("KEYWORD", "NAME"):
                        entity_map[text] = make_tag("F", text, "Function")
                # TYPE NAME ;/= → V#
                elif (idx + 1 < len(raw_tokens)
                      and raw_tokens[idx + 1].lastgroup == "OPERATOR"
                      and raw_tokens[idx + 1].group() in (";", "=", ",", ")")):
                    prev_idx = idx - 1
                    while prev_idx >= 0 and raw_tokens[prev_idx].lastgroup == "WS":
                        prev_idx -= 1
                    if prev_idx >= 0 and raw_tokens[prev_idx].lastgroup in ("KEYWORD", "NAME"):
                        entity_map[text] = make_tag("V", text, "Variable")

        # Pass 2: Build stream
        stream = []
        for m in raw_tokens:
            kind = m.lastgroup
            text = m.group()

            if kind == "NEWLINE":
                stream.append(Coordinate(tag=text, quality="NL", surface=text))
            elif kind == "WS":
                stream.append(Coordinate(tag=text, quality="WS", surface=text))
            elif kind in ("BLOCK_COMMENT", "LINE_COMMENT"):
                stream.append(Coordinate(tag=text, quality="COMMENT", surface=text))
            elif kind in ("INCLUDE", "PREPROC"):
                stream.append(Coordinate(tag=text, quality="KW", surface=text))
            elif kind in ("STRING", "CHAR_LIT"):
                stream.append(Coordinate(tag=text, quality="STR", surface=text))
            elif kind == "NUMBER":
                stream.append(Coordinate(tag=text, quality="LITERAL", surface=text))
            elif kind == "OPERATOR":
                stream.append(Coordinate(tag=text, quality="OP", surface=text))
            elif kind == "KEYWORD":
                stream.append(Coordinate(tag=text, quality="KW", surface=text))
            elif kind == "NAME":
                if text in entity_map:
                    stream.append(Coordinate(tag=entity_map[text], quality="BASE", surface=text))
                else:
                    stream.append(Coordinate(tag=text, quality="KW", surface=text))
            elif kind == "OTHER":
                stream.append(Coordinate(tag=text, quality="OP", surface=text))

        return tag_dict, stream


if __name__ == "__main__":
    tokenizer = TagQualityTokenizer()

    example = "Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed! He said, 'Wow, that is a really amazing vase!'"

    tag_dict, stream = tokenizer.encode(example)

    print("=== 2D Dictionary ===")
    print(tokenizer.format_dictionary(tag_dict))
    print()
    print("=== Coordinate Stream ===")
    print(tokenizer.format_stream(stream))
    print()

    decoded = tokenizer.decode(tag_dict, stream)
    print("=== Decoded ===")
    print(decoded)
    print()
    print(f"=== Lossless: {decoded == example} ===")
    if decoded != example:
        print(f"Original: {repr(example)}")
        print(f"Decoded:  {repr(decoded)}")
