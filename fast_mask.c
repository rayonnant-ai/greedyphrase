/* fast_mask.c — Convert text to masked uint32 word IDs.
 *
 * Two passes over mmap'd text:
 *   1. Build set of lowercase word forms (words starting with lowercase letter)
 *      + pre-load protection list (modal verbs, connectives, prepositions)
 *   2. Classify words, assign uint32 IDs, write binary stream + vocabulary
 *
 * Variable classification (Phase 8c — typed sentinel masking):
 *   - Numbers: starts with digit → SENTINEL_OTHER
 *   - Chemical formulas: H2O, CO2 → SENTINEL_OTHER
 *   - Citation pointers: [1], [23] → SENTINEL_OTHER
 *   - Table/list markers: 1. a) (iv) → SENTINEL_OTHER
 *   - Honorifics: Dr., Prof. → SENTINEL_OTHER
 *   - Compass/directional: North, SSW → SENTINEL_OTHER
 *   - Dialogue: word containing " → SENTINEL_OTHER
 *   - Content adjectives: big, happy, red → SENTINEL_ADJ
 *   - Content nouns: dog, ball, house → SENTINEL_OBJ
 *   - Content verbs: play, eat, said → SENTINEL_ACTION
 *   - Possessive pronouns: his, her, their → SENTINEL_OTHER
 *   - Proper nouns: uppercase, not in lc_set → SENTINEL_PERSON
 *
 * Typed sentinels let the template miner discover type-aware patterns:
 *   "[PERSON] liked the [ADJ] [OBJ]" instead of generic "[?] liked the [?] [?]"
 *
 * Usage: fast_mask <text_file> <words.bin> <vocab.bin>
 *
 * words.bin: uint32 array with LINE_SEP (0xFFFFFFFE) between lines,
 *            typed sentinels (0xFFFFFF00-0xFFFFFF04) for variables.
 * vocab.bin: uint32 num_words, then per word: uint32 len, char[len] bytes
 */
#define XXH_INLINE_ALL
#include "xxhash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define LINE_SEP         0xFFFFFFFEu

/* Typed sentinel IDs */
#define SENTINEL_PERSON  0xFFFFFF00u
#define SENTINEL_OBJ     0xFFFFFF01u
#define SENTINEL_ADJ     0xFFFFFF02u
#define SENTINEL_ACTION  0xFFFFFF03u
#define SENTINEL_OTHER   0xFFFFFF04u
#define SENTINEL_MIN     0xFFFFFF00u
#define IS_SENTINEL(x)   ((x) >= SENTINEL_MIN && (x) < LINE_SEP)

#define SET_BITS    21
#define SET_SIZE    (1U << SET_BITS)
#define SET_MASK    (SET_SIZE - 1)
#define MAP_BITS    21
#define MAP_SIZE    (1U << MAP_BITS)
#define MAP_MASK    (MAP_SIZE - 1)
#define ARENA_BLOCK (64 * 1024 * 1024)
#define OUT_BUF     (4 * 1024 * 1024)

/* ─── Arena ─── */
typedef struct Arena { char *base; size_t used, cap; struct Arena *prev; } Arena;

static Arena *arena_new(void) {
    Arena *a = malloc(sizeof(Arena));
    a->base = malloc(ARENA_BLOCK);
    a->used = 0; a->cap = ARENA_BLOCK; a->prev = NULL;
    return a;
}
static inline void *arena_alloc(Arena **ap, size_t size) {
    size = (size + 7) & ~(size_t)7;
    Arena *a = *ap;
    if (__builtin_expect(a->used + size > a->cap, 0)) {
        Arena *n = malloc(sizeof(Arena));
        n->base = malloc(ARENA_BLOCK); n->used = 0;
        n->cap = ARENA_BLOCK; n->prev = a; *ap = n; a = n;
    }
    void *p = a->base + a->used; a->used += size; return p;
}
static void arena_free_all(Arena *a) {
    while (a) { Arena *p = a->prev; free(a->base); free(a); a = p; }
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─── Lowercase set ─── */
typedef struct LCNode { struct LCNode *next; uint32_t tag; uint16_t len; char s[]; } LCNode;

static LCNode **lc_set;
static Arena *lc_arena;

static void lc_add(const char *key, int len, uint64_t h) {
    uint32_t idx = (uint32_t)h & SET_MASK;
    uint32_t tag = (uint32_t)(h >> 32);
    for (LCNode *nd = lc_set[idx]; nd; nd = nd->next)
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, key, len) == 0)
            return;
    LCNode *nd = arena_alloc(&lc_arena, sizeof(LCNode) + len);
    nd->tag = tag; nd->len = (uint16_t)len;
    memcpy(nd->s, key, len);
    nd->next = lc_set[idx]; lc_set[idx] = nd;
}

static int lc_has(const char *key, int len, uint64_t h) {
    uint32_t idx = (uint32_t)h & SET_MASK;
    uint32_t tag = (uint32_t)(h >> 32);
    for (LCNode *nd = lc_set[idx]; nd; nd = nd->next)
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, key, len) == 0)
            return 1;
    return 0;
}

/* ─── Content word set (typed) ─── */
typedef struct CWNode { struct CWNode *next; uint32_t tag; uint32_t sentinel; uint16_t len; char s[]; } CWNode;

static CWNode **cw_set;
static Arena *cw_arena;

static void cw_add(const char *key, int len, uint64_t h, uint32_t sentinel) {
    uint32_t idx = (uint32_t)h & SET_MASK;
    uint32_t tag = (uint32_t)(h >> 32);
    for (CWNode *nd = cw_set[idx]; nd; nd = nd->next)
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, key, len) == 0)
            return;
    CWNode *nd = arena_alloc(&cw_arena, sizeof(CWNode) + len);
    nd->tag = tag; nd->sentinel = sentinel; nd->len = (uint16_t)len;
    memcpy(nd->s, key, len);
    nd->next = cw_set[idx]; cw_set[idx] = nd;
}

/* Returns typed sentinel ID, or 0 if not found */
static uint32_t cw_lookup(const char *key, int len, uint64_t h) {
    uint32_t idx = (uint32_t)h & SET_MASK;
    uint32_t tag = (uint32_t)(h >> 32);
    for (CWNode *nd = cw_set[idx]; nd; nd = nd->next)
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, key, len) == 0)
            return nd->sentinel;
    return 0;
}

/* ─── Word → ID map ─── */
typedef struct WNode { struct WNode *next; uint32_t tag, id; uint16_t len; char s[]; } WNode;

static WNode **wid_map;
static Arena *wid_arena;
static uint32_t next_id;

/* Vocab array (indexed by ID) for output */
static char **v_words;
static int  *v_lens;
static uint32_t v_cap;

static uint32_t wid_get_or_add(const char *key, int len, uint64_t h) {
    uint32_t idx = (uint32_t)h & MAP_MASK;
    uint32_t tag = (uint32_t)(h >> 32);
    for (WNode *nd = wid_map[idx]; nd; nd = nd->next)
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, key, len) == 0)
            return nd->id;
    uint32_t id = next_id++;
    WNode *nd = arena_alloc(&wid_arena, sizeof(WNode) + len);
    nd->tag = tag; nd->id = id; nd->len = (uint16_t)len;
    memcpy(nd->s, key, len);
    nd->next = wid_map[idx]; wid_map[idx] = nd;
    if (id >= v_cap) {
        v_cap *= 2;
        v_words = realloc(v_words, v_cap * sizeof(char *));
        v_lens  = realloc(v_lens,  v_cap * sizeof(int));
    }
    char *cp = arena_alloc(&wid_arena, len);
    memcpy(cp, key, len);
    v_words[id] = cp; v_lens[id] = len;
    return id;
}

/* ─── Classification helpers ─── */
static inline int is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int is_number(const char *s, int len) {
    if (len == 0 || !isdigit((unsigned char)s[0])) return 0;
    int i = 1;
    while (i < len && (isdigit((unsigned char)s[i]) || s[i] == ',')) i++;
    if (i < len && s[i] == '.') {
        i++;
        while (i < len && isdigit((unsigned char)s[i])) i++;
    }
    return i == len;
}

/* ─── Phase 8: Chemical formula detection ─── */
static int is_chemical(const char *s, int len) {
    if (len < 2 || !isupper((unsigned char)s[0])) return 0;
    int has_digit = 0, has_upper = 0;
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        if (isupper(c)) has_upper = 1;
        else if (isdigit(c)) has_digit = 1;
        else if (islower(c) || c == '(' || c == ')') { /* ok */ }
        else return 0;
    }
    return has_digit && has_upper;
}

/* ─── Phase 8: Citation pointer detection ─── */
static int is_citation(const char *s, int len) {
    if (len < 3 || s[0] != '[' || s[len - 1] != ']') return 0;
    int has_digit = 0;
    for (int i = 1; i < len - 1; i++) {
        unsigned char c = (unsigned char)s[i];
        if (isdigit(c)) has_digit = 1;
        else if (c != ',' && c != '-' && c != ' ') return 0;
    }
    return has_digit;
}

/* ─── Phase 8: Table/list marker detection ─── */
static int is_list_marker(const char *s, int len) {
    if (len < 2 || len > 6) return 0;
    if (isdigit((unsigned char)s[0]) && s[len - 1] == '.') {
        for (int i = 0; i < len - 1; i++)
            if (!isdigit((unsigned char)s[i])) return 0;
        return 1;
    }
    if (islower((unsigned char)s[0]) && (s[len - 1] == ')' || s[len - 1] == '.') && len <= 4) {
        for (int i = 0; i < len - 1; i++)
            if (!islower((unsigned char)s[i])) return 0;
        return 1;
    }
    if (s[0] == '(' && s[len - 1] == ')' && len >= 3) {
        for (int i = 1; i < len - 1; i++)
            if (!islower((unsigned char)s[i])) return 0;
        return 1;
    }
    return 0;
}

/* ─── Phase 8: Honorific detection ─── */
static const char *honorifics[] = {
    "Dr.", "Prof.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.",
    "Lt.", "Col.", "Gen.", "Sgt.", "Cpl.", "Pvt.",
    "Rev.", "Hon.", "Capt.", "Cmdr.", "Adm.", "Maj.",
    "St.", "Gov.", "Pres.", "Sen.", "Rep.",
    NULL
};

static int is_honorific(const char *s, int len) {
    for (const char **p = honorifics; *p; p++)
        if ((int)strlen(*p) == len && memcmp(s, *p, len) == 0)
            return 1;
    return 0;
}

/* ─── Phase 8: Compass/directional detection ─── */
static const char *compass_words[] = {
    "North", "South", "East", "West",
    "Northeast", "Northwest", "Southeast", "Southwest",
    "North-east", "North-west", "South-east", "South-west",
    "NE", "NW", "SE", "SW",
    "NNE", "NNW", "SSE", "SSW", "ENE", "ESE", "WNW", "WSW",
    NULL
};

static int is_compass(const char *s, int len) {
    for (const char **p = compass_words; *p; p++)
        if ((int)strlen(*p) == len && memcmp(s, *p, len) == 0)
            return 1;
    return 0;
}

/* ─── Phase 8b: Dialogue detection ─── */
static int has_quote(const char *s, int len) {
    for (int i = 0; i < len; i++)
        if (s[i] == '"') return 1;
    return 0;
}

/* ─── Phase 8c: Typed content word lists ─── */
static const char *content_adjectives[] = {
    "big", "little", "small", "new", "old", "pretty", "happy", "sad",
    "scared", "brave", "kind", "nice", "mean", "silly", "funny", "curious",
    "gentle", "friendly", "soft", "hard", "bright", "dark", "warm", "cold",
    "sweet", "beautiful", "wonderful", "shiny", "special", "different",
    "favorite", "amazing", "important", "careful", "quiet", "loud", "tall",
    "short", "tiny", "huge", "fast", "slow", "real", "cool", "great",
    "best", "poor", "clean", "dirty", "wet", "dry", "hungry", "tired",
    "angry", "lonely", "excited", "proud",
    /* Phase 8d: temporal/manner adverbs */
    "suddenly", "eventually", "finally", "quickly", "slowly", "carefully",
    "gently", "happily", "sadly", "quietly", "loudly", "bravely", "kindly",
    "softly", "eagerly", "politely", "proudly", "patiently", "cheerfully",
    NULL
};

static const char *content_nouns[] = {
    "boy", "girl", "dog", "cat", "bird", "fish", "bear", "rabbit",
    "monkey", "elephant", "lion", "tiger", "fox", "frog", "mouse", "duck",
    "bunny", "puppy", "kitten", "pony", "ball", "toy", "cake", "apple",
    "flower", "tree", "house", "garden", "park", "forest", "river", "lake",
    "car", "boat", "hat", "dress", "shoe", "book", "box", "bag",
    "cup", "cookie", "candy", "water", "food", "gift", "star", "sun",
    "moon", "rain", "snow", "sand", "rock", "stick", "leaf", "door",
    "bed", "chair", "table", "princess", "prince", "king", "queen",
    "dragon", "fairy", "monster", "robot",
    /* Phase 8d: onomatopoeia / sound words */
    "splash", "crash", "boom", "bang", "buzz", "hiss", "roar", "meow",
    "woof", "quack", "moo", "oink", "chirp", "squeak", "growl", "purr",
    NULL
};

static const char *content_verbs[] = {
    "play", "eat", "run", "build", "hide", "share", "catch", "throw",
    "pull", "push", "hold", "carry", "pick", "open", "close", "break",
    "fix", "sing", "dance", "draw", "paint", "swim", "jump", "climb",
    "fly", "sleep", "cry", "shout", "hug", "kiss", "wave", "kick",
    "bite", "dig", "cook",
    /* Phase 8c: dialogue verbs (attribution) */
    "said", "asked", "replied", "whispered", "shouted", "yelled",
    "called", "told", "answered", "exclaimed",
    NULL
};

/* Phase 8c: possessive pronouns → SENTINEL_OTHER */
static const char *content_possessives[] = {
    "his", "her", "their", "its",
    NULL
};

/* ─── Phase 8: Literal Protection List ─── */
static const char *protected_words[] = {
    /* Modal verbs */
    "could", "should", "would", "can", "may", "might", "shall", "will", "must",
    /* Connectives */
    "however", "therefore", "furthermore", "moreover", "meanwhile",
    "nevertheless", "although", "because", "since", "while", "though",
    "thus", "hence", "yet", "still", "also", "instead", "otherwise",
    "besides", "consequently", "accordingly", "subsequently",
    /* Prepositions */
    "between", "among", "through", "during", "before", "after",
    "above", "below", "across", "against", "along", "around",
    "beneath", "beside", "beyond", "despite", "except", "inside",
    "into", "near", "onto", "outside", "over", "past",
    "toward", "towards", "under", "underneath", "until", "upon",
    "within", "without", "about", "from", "with",
    /* Common function words that appear sentence-initial */
    "there", "here", "where", "when", "what", "which", "who",
    "how", "why", "that", "this", "these", "those",
    "some", "any", "each", "every", "both", "all", "such",
    "many", "much", "most", "more", "other", "another",
    NULL
};

static void preload_protected(void) {
    for (const char **p = protected_words; *p; p++) {
        int len = (int)strlen(*p);
        uint64_t h = XXH3_64bits(*p, len);
        if (!lc_has(*p, len, h))
            lc_add(*p, len, h);
    }
}

static void preload_content_words(void) {
    struct { const char **list; uint32_t sentinel; } typed[] = {
        { content_adjectives,  SENTINEL_ADJ },
        { content_nouns,       SENTINEL_OBJ },
        { content_verbs,       SENTINEL_ACTION },
        { content_possessives, SENTINEL_OTHER },
        { NULL, 0 }
    };
    for (int t = 0; typed[t].list; t++) {
        for (const char **p = typed[t].list; *p; p++) {
            int len = (int)strlen(*p);
            uint64_t h = XXH3_64bits(*p, len);
            cw_add(*p, len, h, typed[t].sentinel);
        }
    }
}

/* ─── Counters for stats ─── */
static uint32_t n_chem = 0, n_cite = 0, n_list = 0, n_honor = 0, n_compass = 0;
static uint32_t n_dialogue = 0, n_cw = 0, n_proper = 0;
static uint32_t n_adj = 0, n_obj = 0, n_act = 0, n_poss = 0;

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <text_file> <words.bin> <vocab.bin>\n", argv[0]);
        return 1;
    }
    double t0 = now_sec();

    /* mmap input */
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st; fstat(fd, &st);
    size_t fsize = (size_t)st.st_size;
    const char *text = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (text == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((void *)text, fsize, MADV_SEQUENTIAL);
    close(fd);

    /* ─── Pass 1: build lowercase vocabulary ─── */
    lc_set = calloc(SET_SIZE, sizeof(LCNode *));
    lc_arena = arena_new();
    char lc_buf[8192];
    uint32_t n_lc = 0;
    size_t pos = 0;

    while (pos < fsize) {
        while (pos < fsize && is_ws(text[pos])) pos++;
        if (pos >= fsize) break;
        size_t ws = pos;
        while (pos < fsize && !is_ws(text[pos])) pos++;
        int wlen = (int)(pos - ws);
        if (wlen > 0 && wlen < 8192 && islower((unsigned char)text[ws])) {
            for (int i = 0; i < wlen; i++)
                lc_buf[i] = (char)tolower((unsigned char)text[ws + i]);
            uint64_t h = XXH3_64bits(lc_buf, wlen);
            if (!lc_has(lc_buf, wlen, h)) {
                lc_add(lc_buf, wlen, h);
                n_lc++;
            }
        }
    }

    /* Pre-load protection list into lowercase set */
    preload_protected();

    /* Build typed content word set */
    cw_set = calloc(SET_SIZE, sizeof(CWNode *));
    cw_arena = arena_new();
    preload_content_words();

    fprintf(stderr, "  %u lowercase types + protected (%.1fs)\n", n_lc, now_sec() - t0);

    /* ─── Pass 2: classify, assign IDs, write binary ─── */
    double t1 = now_sec();
    wid_map = calloc(MAP_SIZE, sizeof(WNode *));
    wid_arena = arena_new();
    next_id = 0;
    v_cap = 512 * 1024;
    v_words = malloc(v_cap * sizeof(char *));
    v_lens  = malloc(v_cap * sizeof(int));

    uint32_t *obuf = malloc(OUT_BUF * sizeof(uint32_t));
    uint32_t obp = 0;
    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror(argv[2]); return 1; }
    char *wbuf = malloc(16 * 1024 * 1024);
    setvbuf(out, wbuf, _IOFBF, 16 * 1024 * 1024);

    uint32_t n_total = 0, n_var = 0, n_lines = 0;
    pos = 0;

    while (pos < fsize) {
        /* Find line boundaries */
        size_t lstart = pos;
        while (pos < fsize && text[pos] != '\n') pos++;
        size_t lend = pos;
        if (pos < fsize) pos++;

        /* Count words in line (skip lines with < 2 words) */
        size_t tp = lstart;
        int nw = 0;
        while (tp < lend) {
            while (tp < lend && is_ws(text[tp])) tp++;
            if (tp >= lend) break;
            while (tp < lend && !is_ws(text[tp])) tp++;
            nw++;
        }
        if (nw < 2) continue;

        /* Ensure output buffer has room (nw words + LINE_SEP) */
        if (obp + (uint32_t)nw + 1 > OUT_BUF) {
            fwrite(obuf, sizeof(uint32_t), obp, out);
            obp = 0;
        }

        /* Process words */
        tp = lstart;
        while (tp < lend) {
            while (tp < lend && is_ws(text[tp])) tp++;
            if (tp >= lend) break;
            size_t ws = tp;
            while (tp < lend && !is_ws(text[tp])) tp++;
            int wlen = (int)(tp - ws);
            n_total++;

            uint32_t sentinel = 0;

            /* Phase 8 extended masks — check specific patterns first */
            if (is_number(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER;
            } else if (is_chemical(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_chem++;
            } else if (is_citation(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_cite++;
            } else if (is_list_marker(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_list++;
            } else if (is_honorific(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_honor++;
            } else if (is_compass(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_compass++;
            } else if (has_quote(text + ws, wlen)) {
                sentinel = SENTINEL_OTHER; n_dialogue++;
            } else {
                /* Content word + proper noun checks need lowercase */
                int clen = wlen < 8192 ? wlen : 8192;
                for (int i = 0; i < clen; i++)
                    lc_buf[i] = (char)tolower((unsigned char)text[ws + i]);
                uint64_t lch = XXH3_64bits(lc_buf, clen);

                uint32_t cw_type = cw_lookup(lc_buf, clen, lch);
                if (cw_type) {
                    sentinel = cw_type;
                    n_cw++;
                    if (cw_type == SENTINEL_ADJ) n_adj++;
                    else if (cw_type == SENTINEL_OBJ) n_obj++;
                    else if (cw_type == SENTINEL_ACTION) n_act++;
                    else n_poss++;
                } else if (wlen > 1 && isupper((unsigned char)text[ws])) {
                    if (!lc_has(lc_buf, clen, lch)) {
                        sentinel = SENTINEL_PERSON;
                        n_proper++;
                    }
                }
            }

            if (sentinel) {
                obuf[obp++] = sentinel;
                n_var++;
            } else {
                uint64_t h = XXH3_64bits(text + ws, wlen);
                obuf[obp++] = wid_get_or_add(text + ws, wlen, h);
            }
        }
        obuf[obp++] = LINE_SEP;
        n_lines++;
    }
    if (obp > 0) fwrite(obuf, sizeof(uint32_t), obp, out);
    fclose(out);

    size_t binsz = 0;
    { struct stat s2; if (stat(argv[2], &s2) == 0) binsz = s2.st_size; }
    fprintf(stderr, "  %u words, %u variable (%.1f%%)\n",
            n_total, n_var, 100.0 * n_var / n_total);
    fprintf(stderr, "  Masks: chem=%u cite=%u list=%u honor=%u compass=%u dialogue=%u\n",
            n_chem, n_cite, n_list, n_honor, n_compass, n_dialogue);
    fprintf(stderr, "  Typed: adj=%u obj=%u action=%u poss=%u proper=%u\n",
            n_adj, n_obj, n_act, n_poss, n_proper);
    fprintf(stderr, "  %u lines, %u unique words\n", n_lines, next_id);
    fprintf(stderr, "  Wrote %s (%zu MB, %.1fs)\n", argv[2], binsz >> 20, now_sec() - t1);

    /* ─── Write vocab.bin ─── */
    FILE *vf = fopen(argv[3], "wb");
    if (!vf) { perror(argv[3]); return 1; }
    fwrite(&next_id, 4, 1, vf);
    for (uint32_t i = 0; i < next_id; i++) {
        uint32_t wl = (uint32_t)v_lens[i];
        fwrite(&wl, 4, 1, vf);
        fwrite(v_words[i], 1, wl, vf);
    }
    fclose(vf);
    fprintf(stderr, "  Total: %.1fs\n", now_sec() - t0);

    munmap((void *)text, fsize);
    free(lc_set); arena_free_all(lc_arena);
    free(cw_set); arena_free_all(cw_arena);
    free(wid_map); arena_free_all(wid_arena);
    free(obuf); free(wbuf); free(v_words); free(v_lens);
    return 0;
}
