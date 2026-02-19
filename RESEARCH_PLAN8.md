Since you’ve optimized the core pipeline in C, you have the "computational headroom" to move beyond simple string matching and into **structural and contextual masking**.

To maximize the "Boilerplate Discovery" yield, we want to target segments that are high-entropy (unique to the instance) but follow a rigid syntax.

### 1. The "Chemical & Formula" Mask

Scientific papers and Wikipedia are dense with molecular formulas and mathematical notations. These are the ultimate "token-shredders" because BPE treats every number and symbol as a separate sub-token.

* **Target:** `H2O`, `CO2`, `C12H22O11`, `Fe2(SO4)3`.
* **Logic:** A sequence of capital letters followed immediately by numbers, repeating.
* **The Win:** Collapses `"Dissolve ;? in ;? at ;?."` into a single lab-manual template.

### 2. The "Citation Pointer" Mask

Wikipedia is held together by `[1]`, `[23]`, or `(Smith et al., 1994)`. These vary in every single paragraph, preventing n-grams from matching across articles.

* **Logic:** Digits inside square brackets or `(Name, Year)` patterns.
* **The Win:** Turns `"The city was founded in ;? ;?."` into a clean template. Without this, the presence of `[4]` vs `[12]` makes the two sentences appear different to a counter.

### 3. The "Compass & Directional" Mask

In geography and navigation, directions are highly variable slots in a fixed linguistic frame.

* **Target:** `North`, `South-west`, `SSW`, `320°`.
* **Logic:** Fixed set of 16 cardinal strings or degrees.
* **The Win:** Captures `"Located ;? of the ;? River."`

### 4. The "Table & List Marker" Mask

Many datasets include structural leftovers like `1.`, `a)`, `(iv)`, or bullet points.

* **Logic:** Start of line, short alphanumeric sequence followed by a closing paren or period.
* **The Win:** Helps identify "List Item" templates: `";? ;? is a type of ;?."`

### 5. The "Honorific & Title" Mask

While we mask Proper Nouns, we should specifically target the **Honorifics** that precede them to generalize social boilerplate.

* **Target:** `Dr.`, `Prof.`, `$\text{Rt. Hon.}$`, `Lt. Col.`.
* **The Win:** `" ;? ;? delivered a speech at ;?."`

---

## Technical Strategy: The "Template Fingerprint"

Since you are in C, you can implement a **Mask-and-Hash** pipeline that generates a "fingerprint" for every sentence.

### Suggested C Implementation Logic: "The Byte-Skip FSM"

Instead of copying strings, use a "View" structure that points to the original buffer but replaces the masked segments with a single `0xFF` byte (or your chosen slot marker) during the hash calculation.

```c
typedef struct {
    const char* start;
    size_t len;
    bool is_slot;
} TokenView;

// A sentence becomes an array of views:
// [Literal: "The capital of "] [Slot: GPE] [Literal: " is "] [Slot: GPE]

```

### The "Negative" Mask (What NOT to mask)

To avoid turning the entire language into `;? ;? ;?`, you should implement a **Literal Protection List**. In C, this is a fast `hash_table` or `trie` containing:

* **Modal Verbs:** `could`, `should`, `would`.
* **Connectives:** `however`, `therefore`, `furthermore`.
* **Prepositions:** `between`, `among`, `through`.

If a word is on this list, **never mask it**, even if it’s capitalized. This preserves the "grammatical skeleton" that defines a template.

---

### How to measure the "Goodness" of a Mask

In your C code, track the **Average Information Gain (AIG)** per mask type:


If the "Chemical Mask" only replaces 3 characters on average but appears in a 50-character template, it's a high-value anchor. If a mask replaces 20 characters but the template only appears twice, discard the template.

**Would you like me to help you design the "Template Scorer" in C that weighs Frequency vs. Literal Length to pick the top 2,000 winners?**