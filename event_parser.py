"""Event Model Parser for TinyStories (spaCy-based).

Converts raw prose into structured event model text:
  @Ben:person(little,boy) @Store:place(store)
  Ben love play
  Ben go_to Park
  Ben see butterfly Garden

Pipeline: text -> EventParser.transform() -> event_model_text -> byte_tokenizer

Requires: python3.10, spacy, en_core_web_sm
"""

import re
from dataclasses import dataclass, field

import spacy


@dataclass
class Entity:
    name: str
    etype: str  # person, animal, place, object
    descriptors: list = field(default_factory=list)
    gender: str = "n"  # m, f, n


@dataclass
class Event:
    agent: str
    action: str
    args: list = field(default_factory=list)


# ── Type Words ──
TYPE_WORDS = {}
for w in ("boy", "man", "king", "prince", "father", "dad", "brother",
          "uncle", "grandpa", "grandfather", "son", "husband"):
    TYPE_WORDS[w] = ("person", "m")
for w in ("girl", "woman", "queen", "princess", "mother", "mom", "sister",
          "aunt", "grandma", "grandmother", "daughter", "wife"):
    TYPE_WORDS[w] = ("person", "f")
for w in ("child", "kid", "baby", "friend", "person", "teacher",
          "doctor", "farmer", "shopkeeper", "owner", "neighbor",
          "firefighter", "pirate", "wizard", "knight", "clown",
          "sailor", "hunter", "otter"):
    TYPE_WORDS[w] = ("person", "n")
for w in ("cat", "dog", "bird", "fish", "bunny", "rabbit", "bear",
          "lion", "monkey", "mouse", "frog", "duck", "turtle", "horse",
          "cow", "pig", "sheep", "chicken", "elephant", "fox", "deer",
          "squirrel", "butterfly", "bee", "ant", "spider", "worm",
          "dinosaur", "dragon", "puppy", "kitten", "tiger"):
    TYPE_WORDS[w] = ("animal", "n")
for w in ("store", "park", "school", "house", "home", "garden", "forest",
          "beach", "lake", "river", "mountain", "hill", "field", "farm",
          "village", "town", "city", "castle", "cave", "island",
          "playground", "zoo", "library", "hospital", "kitchen",
          "bedroom", "yard", "pond", "ocean", "market", "shop",
          "bakery", "room", "street", "road", "bridge", "tower"):
    TYPE_WORDS[w] = ("place", "n")
for w in ("toy", "ball", "doll", "book", "cake", "cookie", "flower",
          "tree", "rock", "star", "hat", "shoe", "dress", "shirt",
          "key", "door", "box", "bag", "cup", "bottle", "car", "boat",
          "bike", "kite", "balloon", "swing", "slide", "blanket",
          "pillow", "lamp", "picture", "mirror", "candy", "apple",
          "banana", "sandwich", "present", "gift", "coin",
          "treasure", "sword", "crown", "wand", "map", "rope", "basket",
          "nest", "shell", "feather", "seed", "leaf", "stick", "stone",
          "ring", "necklace", "bracelet", "vase", "jar", "pot", "pan",
          "water", "food", "ice cream"):
    TYPE_WORDS[w] = ("object", "n")

COMMON_ADJS = {
    "little", "big", "small", "old", "young", "new", "good", "bad",
    "happy", "sad", "pretty", "beautiful", "ugly", "tall", "short",
    "long", "kind", "mean", "nice", "brave", "scared", "angry",
    "hungry", "thirsty", "tired", "sleepy", "silly", "funny",
    "special", "amazing", "wonderful", "terrible", "best", "worst",
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "pink", "purple", "orange", "golden", "dark", "bright", "shiny",
    "soft", "hard", "warm", "cold", "hot", "cool", "wet", "dry",
    "fast", "slow", "loud", "quiet", "strong", "weak", "smart",
    "sweet", "sour", "delicious", "yummy", "favorite", "certain",
    "other", "strange", "curious", "gentle", "poor", "rich",
    "magical", "tiny", "huge", "enormous", "lovely", "different",
    "friendly", "fierce", "reliable", "smooth",
}

# Named introduction patterns
_INTRO_NAMED = re.compile(
    r'(?:there was|there lived|there is)\s+a[n]?\s+([\w\s,]+?)\s+named\s+(\w+)',
    re.IGNORECASE,
)
_INTRO_WAS = re.compile(
    r'(\w+)\s+was\s+a[n]?\s+([\w\s,]+)',
    re.IGNORECASE,
)

SKIP_WORDS = {
    "the", "a", "an", "and", "or", "but", "so", "then", "very",
    "too", "really", "just", "also", "only", "not", "no", "yes",
    "one", "two", "three", "all", "some", "many", "much", "more",
    "most", "every", "each", "this", "that", "these", "those",
    "my", "your", "our", "their", "its", "his", "her",
    "once", "upon", "time", "day", "way",
    "there", "here", "now", "it", "they", "we", "i",
    "if", "because", "while", "after", "before", "until", "since",
    "would", "could", "should", "might", "will", "can", "may",
    "do", "did", "does",
    "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "into", "through", "over", "under", "up", "down", "out", "off",
    "away", "back", "around", "along", "across", "between",
}

# ── Reconstruction Templates ──
EVENT_TEMPLATES = {
    "be":       "{agent} was {arg_str}.",
    "go":       "{agent} went to {arg_str}.",
    "see":      "{agent} saw {arg_str}.",
    "say":      '{agent} said, "{arg_str}."',
    "tell":     "{agent} told {arg_str}.",
    "ask":      "{agent} asked {arg_str}.",
    "take":     "{agent} took {arg_str}.",
    "give":     "{agent} gave {arg_str}.",
    "bring":    "{agent} brought {arg_str}.",
    "have":     "{agent} had {arg_str}.",
    "like":     "{agent} liked {arg_str}.",
    "love":     "{agent} loved {arg_str}.",
    "want":     "{agent} wanted {arg_str}.",
    "hope":     "{agent} hoped {arg_str}.",
    "play":     "{agent} played {arg_str}.",
    "help":     "{agent} helped {arg_str}.",
    "make":     "{agent} made {arg_str}.",
    "find":     "{agent} found {arg_str}.",
    "get":      "{agent} got {arg_str}.",
    "put":      "{agent} put {arg_str}.",
    "eat":      "{agent} ate {arg_str}.",
    "drink":    "{agent} drank {arg_str}.",
    "sleep":    "{agent} fell asleep.",
    "wake":     "{agent} woke up.",
    "sit":      "{agent} sat down.",
    "stand":    "{agent} stood up.",
    "fall":     "{agent} fell {arg_str}.",
    "jump":     "{agent} jumped {arg_str}.",
    "climb":    "{agent} climbed {arg_str}.",
    "fly":      "{agent} flew {arg_str}.",
    "swim":     "{agent} swam {arg_str}.",
    "run":      "{agent} ran {arg_str}.",
    "walk":     "{agent} walked {arg_str}.",
    "open":     "{agent} opened {arg_str}.",
    "close":    "{agent} closed {arg_str}.",
    "break":    "{agent} broke {arg_str}.",
    "fix":      "{agent} fixed {arg_str}.",
    "clean":    "{agent} cleaned {arg_str}.",
    "try":      "{agent} tried {arg_str}.",
    "start":    "{agent} started {arg_str}.",
    "stop":     "{agent} stopped {arg_str}.",
    "finish":   "{agent} finished {arg_str}.",
    "learn":    "{agent} learned {arg_str}.",
    "teach":    "{agent} taught {arg_str}.",
    "read":     "{agent} read {arg_str}.",
    "write":    "{agent} wrote {arg_str}.",
    "sing":     "{agent} sang {arg_str}.",
    "dance":    "{agent} danced {arg_str}.",
    "smile":    "{agent} smiled.",
    "laugh":    "{agent} laughed.",
    "cry":      "{agent} cried.",
    "hug":      "{agent} hugged {arg_str}.",
    "thank":    "{agent} thanked {arg_str}.",
    "feel":     "{agent} felt {arg_str}.",
    "know":     "{agent} knew {arg_str}.",
    "think":    "{agent} thought {arg_str}.",
    "remember": "{agent} remembered {arg_str}.",
    "forget":   "{agent} forgot {arg_str}.",
    "believe":  "{agent} believed {arg_str}.",
    "decide":   "{agent} decided {arg_str}.",
    "need":     "{agent} needed {arg_str}.",
    "use":      "{agent} used {arg_str}.",
    "promise":  "{agent} promised {arg_str}.",
    "agree":    "{agent} agreed {arg_str}.",
    "wait":     "{agent} waited {arg_str}.",
    "stay":     "{agent} stayed {arg_str}.",
    "live":     "{agent} lived {arg_str}.",
    "grow":     "{agent} grew {arg_str}.",
    "become":   "{agent} became {arg_str}.",
    "lose":     "{agent} lost {arg_str}.",
    "win":      "{agent} won {arg_str}.",
    "fight":    "{agent} fought {arg_str}.",
    "hit":      "{agent} hit {arg_str}.",
    "kick":     "{agent} kicked {arg_str}.",
    "throw":    "{agent} threw {arg_str}.",
    "catch":    "{agent} caught {arg_str}.",
    "pull":     "{agent} pulled {arg_str}.",
    "push":     "{agent} pushed {arg_str}.",
    "carry":    "{agent} carried {arg_str}.",
    "hold":     "{agent} held {arg_str}.",
    "drop":     "{agent} dropped {arg_str}.",
    "let":      "{agent} let {arg_str}.",
    "show":     "{agent} showed {arg_str}.",
    "hide":     "{agent} hid {arg_str}.",
    "keep":     "{agent} kept {arg_str}.",
    "leave":    "{agent} left {arg_str}.",
    "move":     "{agent} moved {arg_str}.",
    "follow":   "{agent} followed {arg_str}.",
    "lead":     "{agent} led {arg_str}.",
    "send":     "{agent} sent {arg_str}.",
    "meet":     "{agent} met {arg_str}.",
    "invite":   "{agent} invited {arg_str}.",
    "visit":    "{agent} visited {arg_str}.",
    "miss":     "{agent} missed {arg_str}.",
    "save":     "{agent} saved {arg_str}.",
    "scare":    "{agent} was scared of {arg_str}.",
    "surprise": "{agent} was surprised by {arg_str}.",
    "worry":    "{agent} was worried about {arg_str}.",
    "care":     "{agent} cared about {arg_str}.",
    "hurt":     "{agent} hurt {arg_str}.",
    "look":     "{agent} looked at {arg_str}.",
    "call":     "{agent} called {arg_str}.",
    "share":    "{agent} shared {arg_str}.",
    "pick":     "{agent} picked {arg_str}.",
    "turn":     "{agent} turned {arg_str}.",
    "come":     "{agent} came {arg_str}.",
    "bark":     "{agent} barked.",
    "shake":    "{agent} shook {arg_str}.",
    "happen":   "{arg_str} happened.",
    "wear":     "{agent} wore {arg_str}.",
    "build":    "{agent} built {arg_str}.",
    "draw":     "{agent} drew {arg_str}.",
    "fill":     "{agent} filled {arg_str}.",
    "set":      "{agent} set {arg_str}.",
}


class EventParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        self.nlp.max_length = 2_000_000  # handle big stories

    def transform(self, text, story_sep="<|endoftext|>"):
        """Transform a multi-story text into event model text."""
        stories = text.split(story_sep)
        results = []
        n = len(stories)
        for i, story in enumerate(stories):
            story = story.strip()
            if not story:
                continue
            event_text = self.transform_story(story)
            if event_text:
                results.append(event_text)
            if i % 5000 == 0 and i > 0:
                print(f"  {i}/{n} stories processed...", flush=True)
        return (story_sep + "\n").join(results)

    def transform_story(self, story):
        """Convert a single story to event model text."""
        story = story.strip()
        if not story:
            return ""

        # Extract entities from the raw text (regex-based, fast)
        entities = self._extract_entities(story)
        if not entities:
            return story

        # Build entity lookup: lowercase noun -> entity name
        entity_nouns = {}  # "ben" -> "Ben", "park" -> "Park"
        for name, ent in entities.items():
            entity_nouns[name.lower()] = name
            for d in ent.descriptors:
                if d in TYPE_WORDS:
                    entity_nouns[d] = name

        # Parse with spaCy
        doc = self.nlp(story)

        events = []
        # Track last-mentioned entities for pronoun resolution
        last_male = None
        last_female = None
        last_neutral = None
        for name, ent in entities.items():
            if ent.gender == "m":
                last_male = name
            elif ent.gender == "f":
                last_female = name
            else:
                last_neutral = name

        for sent in doc.sents:
            sent_events = self._extract_events_from_sent(
                sent, entities, entity_nouns,
                last_male, last_female, last_neutral,
            )
            for ev, lm, lf, ln in sent_events:
                events.append(ev)
                last_male, last_female, last_neutral = lm, lf, ln

        if not events:
            return story

        return self._render_story(entities, events)

    def _resolve_pronoun(self, tok, entities, last_male, last_female, last_neutral, last_agent=None):
        """Resolve a pronoun token to an entity name, or return None."""
        lower = tok.text.lower()
        if lower in ("he", "him", "his", "himself") and last_male:
            return last_male
        if lower in ("she", "her", "hers", "herself") and last_female:
            return last_female
        if lower in ("it", "its", "itself") and last_neutral:
            return last_neutral
        if lower in ("they", "them", "their", "themselves") and last_agent:
            return last_agent
        return None

    def _tok_to_entity(self, tok, entities, entity_nouns, last_male, last_female, last_neutral, last_agent=None):
        """Map a spaCy token to an entity name, or return None."""
        # Direct name match
        if tok.text in entities:
            return tok.text
        # Pronoun resolution
        if tok.pos_ == "PRON":
            return self._resolve_pronoun(tok, entities, last_male, last_female, last_neutral, last_agent)
        # Noun match via lemma
        lemma = tok.lemma_.lower()
        if lemma in entity_nouns:
            return entity_nouns[lemma]
        # Text match
        if tok.text.lower() in entity_nouns:
            return entity_nouns[tok.text.lower()]
        return None

    def _get_noun_phrase(self, tok, entities, entity_nouns, last_male, last_female, last_neutral, last_agent=None):
        """Get the text for a noun argument. Returns entity name if possible, else noun phrase."""
        ent_name = self._tok_to_entity(tok, entities, entity_nouns,
                                        last_male, last_female, last_neutral, last_agent)
        if ent_name:
            return ent_name

        # Use lemma as a compact representation
        return tok.lemma_.lower()

    def _extract_events_from_sent(self, sent, entities, entity_nouns,
                                   last_male, last_female, last_neutral):
        """Extract events from a spaCy Span using dependency parse.

        Returns list of (Event, last_male, last_female, last_neutral).
        """
        results = []
        last_agent = last_male or last_female or last_neutral

        # Find top-level verbs (skip xcomp — we'll inline those)
        verbs = []
        for tok in sent:
            if tok.pos_ == "VERB" and tok.dep_ in ("ROOT", "conj", "advcl", "ccomp"):
                verbs.append(tok)
            if tok.pos_ == "AUX" and tok.dep_ == "ROOT":
                verbs.append(tok)

        for verb in verbs:
            action = verb.lemma_.lower()

            # Find subject
            agent_name = None
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    agent_name = self._tok_to_entity(
                        child, entities, entity_nouns,
                        last_male, last_female, last_neutral, last_agent,
                    )
                    break

            # For conjoined verbs without explicit subject, inherit from previous
            if not agent_name and verb.dep_ == "conj":
                agent_name = last_agent

            if not agent_name:
                continue

            # Collect arguments
            args = []

            for child in verb.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    args.append(self._get_noun_phrase(
                        child, entities, entity_nouns,
                        last_male, last_female, last_neutral, last_agent,
                    ))
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            np = self._get_noun_phrase(
                                pobj, entities, entity_nouns,
                                last_male, last_female, last_neutral, last_agent,
                            )
                            # Keep preposition for non-entity nouns
                            prep = child.text.lower()
                            if np in entities:
                                args.append(np)
                            else:
                                args.append(f"{prep} {np}")
                elif child.dep_ == "dative":
                    args.append(self._get_noun_phrase(
                        child, entities, entity_nouns,
                        last_male, last_female, last_neutral, last_agent,
                    ))
                elif child.dep_ == "acomp":
                    args.append(child.text.lower())
                elif child.dep_ == "xcomp":
                    # Inline xcomp: "loved to play" -> love play
                    if child.pos_ == "VERB":
                        xcomp_args = [child.lemma_.lower()]
                        # Get xcomp's own objects
                        for xchild in child.children:
                            if xchild.dep_ in ("dobj", "attr"):
                                xcomp_args.append(self._get_noun_phrase(
                                    xchild, entities, entity_nouns,
                                    last_male, last_female, last_neutral, last_agent,
                                ))
                            elif xchild.dep_ == "prep":
                                for pobj in xchild.children:
                                    if pobj.dep_ == "pobj":
                                        np = self._get_noun_phrase(
                                            pobj, entities, entity_nouns,
                                            last_male, last_female, last_neutral, last_agent,
                                        )
                                        if np in entities:
                                            xcomp_args.append(np)
                                        else:
                                            xcomp_args.append(f"{xchild.text.lower()} {np}")
                        args.extend(xcomp_args)
                    else:
                        args.append(child.lemma_.lower())

            # Build event
            event = Event(agent=agent_name, action=action, args=args)

            # Update coreference tracking
            agent_ent = entities.get(agent_name)
            last_agent = agent_name
            if agent_ent:
                if agent_ent.gender == "m":
                    last_male = agent_name
                elif agent_ent.gender == "f":
                    last_female = agent_name
                else:
                    last_neutral = agent_name

            results.append((event, last_male, last_female, last_neutral))

        return results

    def reconstruct(self, event_text, story_sep="<|endoftext|>"):
        """Convert event model text back to readable prose."""
        stories = event_text.split(story_sep)
        results = []
        for story_block in stories:
            story_block = story_block.strip()
            if not story_block:
                continue
            results.append(self._reconstruct_story(story_block))
        return (" " + story_sep + " ").join(results)

    def _reconstruct_story(self, story_block):
        """Reconstruct a single story block to prose."""
        lines = story_block.strip().split("\n")
        if not lines:
            return ""

        # Parse header
        entities = {}
        header_line = lines[0]
        for match in re.finditer(r'@(\w+):(\w+)\(([^)]*)\)', header_line):
            name, etype, desc_str = match.group(1), match.group(2), match.group(3)
            descriptors = [d.strip() for d in desc_str.split(",") if d.strip()]
            entities[name] = Entity(name=name, etype=etype, descriptors=descriptors)

        prose_parts = []
        intro_done = set()  # entities that have been introduced
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            agent = parts[0]
            action = parts[1]
            args = parts[2:]
            arg_str = " ".join(args) if args else ""

            ent = entities.get(agent)

            # First mention of an entity with "be" -> introduction
            if action == "be" and ent and agent not in intro_done:
                desc = " ".join(ent.descriptors[:-1]) if len(ent.descriptors) > 1 else ""
                tword = ent.descriptors[-1] if ent.descriptors else ent.etype
                if not intro_done:  # very first entity
                    if desc:
                        prose_parts.append(f"Once upon a time, there was a {desc} {tword} named {agent}.")
                    else:
                        prose_parts.append(f"Once upon a time, there was a {tword} named {agent}.")
                else:
                    if desc:
                        prose_parts.append(f"There was also a {desc} {tword} named {agent}.")
                    else:
                        prose_parts.append(f"There was also a {tword} named {agent}.")
                intro_done.add(agent)
                # If there are args (e.g. "be happy"), append them
                if args:
                    prose_parts.append(f"{agent} was {arg_str}.")
                continue

            template = EVENT_TEMPLATES.get(action)
            if template:
                fmt = {"agent": agent, "arg_str": arg_str if arg_str else ""}
                try:
                    result = template.format(**fmt)
                except (KeyError, IndexError):
                    result = f"{agent} {action} {arg_str}."
            else:
                result = f"{agent} {action} {arg_str}."

            # Clean up artifacts
            result = result.replace("  ", " ").strip()
            if result.endswith(" ."):
                result = result[:-2] + "."
            if not result.endswith(".") and not result.endswith('"'):
                result += "."
            prose_parts.append(result)

        return " ".join(prose_parts)

    def _extract_entities(self, text):
        """Extract entities from story text using regex cascades."""
        entities = {}

        # Pass 1: Named introductions
        for match in _INTRO_NAMED.finditer(text):
            desc_str = match.group(1).strip()
            name = match.group(2)
            words = desc_str.split()
            adjs = []
            type_word = None
            gender = "n"
            for w in words:
                wl = w.lower().strip(",")
                if wl in TYPE_WORDS:
                    type_word = wl
                    _, gender = TYPE_WORDS[wl]
                elif wl in COMMON_ADJS:
                    adjs.append(wl)
            if type_word:
                descriptors = adjs + [type_word]
                etype = TYPE_WORDS[type_word][0]
            else:
                descriptors = adjs if adjs else ["character"]
                etype = "person"
            entities[name] = Entity(name=name, etype=etype,
                                    descriptors=descriptors, gender=gender)

        # Pass 2: "[Name] was a [adj] [type]"
        for match in _INTRO_WAS.finditer(text):
            name = match.group(1)
            if name.lower() in SKIP_WORDS or name in entities:
                continue
            if not name[0].isupper():
                continue
            desc_str = match.group(2).strip()
            words = desc_str.split()
            adjs = []
            type_word = None
            gender = "n"
            for w in words[:5]:
                wl = w.lower().strip(",.")
                if wl in TYPE_WORDS:
                    type_word = wl
                    _, gender = TYPE_WORDS[wl]
                    break
                elif wl in COMMON_ADJS:
                    adjs.append(wl)
                else:
                    break
            if type_word and name not in entities:
                descriptors = adjs + [type_word]
                etype = TYPE_WORDS[type_word][0]
                entities[name] = Entity(name=name, etype=etype,
                                        descriptors=descriptors, gender=gender)

        # Pass 3: Capitalized name scan
        words = text.split()
        cap_counts = {}
        for i, w in enumerate(words):
            clean = w.strip(".,!?;:\"'()-")
            if not clean or not clean[0].isupper():
                continue
            if clean.lower() in SKIP_WORDS:
                continue
            if i > 0:
                prev = words[i - 1]
                if prev and prev[-1] in ".!?":
                    continue
            if clean not in entities:
                cap_counts[clean] = cap_counts.get(clean, 0) + 1

        for name, count in cap_counts.items():
            if count >= 2 and name not in entities:
                nl = name.lower()
                if nl in TYPE_WORDS:
                    etype, gender = TYPE_WORDS[nl]
                    entities[name] = Entity(name=name, etype=etype,
                                            descriptors=[nl], gender=gender)
                else:
                    entities[name] = Entity(name=name, etype="person",
                                            descriptors=["character"], gender="n")

        # Pass 4: Unnamed entities — "the park", "the shopkeeper"
        for match in re.finditer(r'\bthe\s+(\w+)\b', text, re.IGNORECASE):
            w = match.group(1).lower()
            if w in TYPE_WORDS:
                canon = w.capitalize()
                if canon not in entities:
                    etype, gender = TYPE_WORDS[w]
                    entities[canon] = Entity(name=canon, etype=etype,
                                             descriptors=[w], gender=gender)

        return entities

    def _render_story(self, entities, events):
        """Render entity declarations + events as event model text."""
        parts = []

        # Header
        decls = []
        for name, ent in entities.items():
            desc = ",".join(ent.descriptors) if ent.descriptors else ent.etype
            decls.append(f"@{name}:{ent.etype}({desc})")
        parts.append(" ".join(decls))

        # Events
        for event in events:
            line = f"{event.agent} {event.action}"
            if event.args:
                line += " " + " ".join(event.args)
            parts.append(line)

        return "\n".join(parts)
