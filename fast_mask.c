/* fast_mask.c — Convert text to masked uint32 word IDs.
 *
 * Two passes over mmap'd text:
 *   1. Build set of lowercase word forms (words starting with lowercase letter)
 *   2. Classify words, assign uint32 IDs, write binary stream + vocabulary
 *
 * Variable classification (matches Python is_variable):
 *   - Numbers: starts with digit, rest is digits/commas/periods
 *   - Proper nouns: length > 1, starts uppercase, lowercase form not in set
 *
 * Usage: fast_mask <text_file> <words.bin> <vocab.bin>
 *
 * words.bin: uint32 array with LINE_SEP (0xFFFFFFFE) between lines,
 *            SENTINEL (0xFFFFFFFF) for variables.
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

#define LINE_SEP    0xFFFFFFFEu
#define SENTINEL_ID 0xFFFFFFFFu
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

/* ─── Classification ─── */
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
    fprintf(stderr, "  %u lowercase types (%.1fs)\n", n_lc, now_sec() - t0);

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

            int var = 0;
            if (is_number(text + ws, wlen)) {
                var = 1;
            } else if (wlen > 1 && isupper((unsigned char)text[ws])) {
                for (int i = 0; i < wlen && i < 8192; i++)
                    lc_buf[i] = (char)tolower((unsigned char)text[ws + i]);
                uint64_t h = XXH3_64bits(lc_buf, wlen < 8192 ? wlen : 8192);
                if (!lc_has(lc_buf, wlen < 8192 ? wlen : 8192, h))
                    var = 1;
            }

            if (var) {
                obuf[obp++] = SENTINEL_ID;
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
    free(wid_map); arena_free_all(wid_arena);
    free(obuf); free(wbuf); free(v_words); free(v_lens);
    return 0;
}
