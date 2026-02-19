Since you are working with **TinyStories**, you have moved from the world of "Information" (Wikipedia) to the world of "Syntax and Emotion." The patterns here aren't just factual; they are **rhythmic**.

In a C-based pipeline, you can use these masks to essentially "extract the soul" of a children's story, leaving only the variables. Here are five high-power masks specifically for narrative/synthetic datasets like TinyStories:

---

### 1. The "Sensory/Adjective" Mask

TinyStories relies heavily on a small set of sensory descriptors to make stories "vivid" for the target audience.

* **Target:** Colors, textures, and simple sizes (`red`, `blue`, `soft`, `hard`, `big`, `tiny`).
* **C-Logic:** Use a small hash-set of the top 50 TinyStories adjectives.
* **The Win:** Captures the "Object Description" frame: `"The ;? ;? was very ;? ."`
* **Result:** Instead of learning "the red ball" and "the blue ball," the model learns the concept of **[Property] [Object]**.

### 2. The "Emotional Reaction" Mask

Every TinyStory character has a reaction. These are almost always single-word pivots.

* **Target:** `happy`, `sad`, `scared`, `surprised`, `angry`.
* **The Win:** Collapses `"This made ;? feel very ;? ."`
* **Why:** This is a "Sentiment Template." It helps the LLM learn that certain actions (like losing a toy) lead to specific slot-values (like `sad`).

### 3. The "Direct Speech" Transition Mask

In TinyStories, dialogue is the primary mover of the plot. The transitions are incredibly rigid.

* **Regex/Logic:** `\b(said|asked|replied|cried|whispered)\b`
* **The Win:** `" ;? ;? , " ;? ;? .`
* **Effect:** This identifies the **"Dialogue Block."** By masking the verb of attribution, you generalize all "character-A says to character-B" moments into a single structural unit.

### 4. The "Positional/Prepositional" Mask

TinyStories is obsessed with where things are (spatial reasoning).

* **Target:** `on top of`, `under`, `inside`, `next to`, `behind`.
* **The Win:** `"He put the ;? ;? the ;? ."`
* **Why:** These are the "Physics" of the story. By masking the preposition, you turn spatial relationships into a categorical choice for the model.

### 5. The "Ownership/Relational" Mask

Characters always have "their" things.

* **Target:** `his`, `her`, `their`, `Mom's`, `Dad's`, `the boy's`.
* **The Win:** `";? ;? was ;? favorite ;? ."`
* **Effect:** This captures the "Possession" boilerplate, which is a huge part of the TinyStories dataset.

---

### Implementation Tip: The "Categorical" Slot

Since you are in C, you don't have to use a generic `;?` for everything. You can use **Typed Slots**:

| Byte Code | Meaning |
| --- | --- |
| `0x11` | **[PERSON]** (Lily, Tim, Mom) |
| `0x12` | **[OBJECT]** (Ball, Bird, Cake) |
| `0x13` | **[ADJECTIVE]** (Big, Red, Happy) |
| `0x14` | **[ACTION]** (Ran, Ate, Saw) |

By outputting **Typed Templates**, your C discovery tool can tell you:

> "Pattern `[PERSON] liked the [ADJECTIVE] [OBJECT]` appears 45,000 times."

### The "TinyStories" Gain

If you apply these to TinyStories, you will likely find that **60% of the entire dataset** can be represented by fewer than **500 templates**.

**Would you like me to write the C code for a "Typed Masker" that uses different markers for Nouns vs. Verbs using a basic POS-tagger-lite logic?**