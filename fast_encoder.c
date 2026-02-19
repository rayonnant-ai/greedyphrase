#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <immintrin.h>

#define MAX_TOKEN_LEN 4096
#define OUTPUT_BUF_TOKENS (512 * 1024)  /* 512K tokens = 1MB buffer */
#define INITIAL_POOL_CAP (1 << 20)      /* 1M nodes initial capacity */

/* --- Trie with contiguous node pool --- */

typedef struct {
    int32_t children[256];  /* index into pool, -1 = no child */
    int32_t token_id;       /* -1 if not a token end */
} TrieNode;

static TrieNode *pool;
static int32_t pool_size;
static int32_t pool_cap;

static int32_t alloc_node(void) {
    if (pool_size >= pool_cap) {
        pool_cap *= 2;
        pool = realloc(pool, (size_t)pool_cap * sizeof(TrieNode));
        if (!pool) { fprintf(stderr, "Out of memory\n"); exit(1); }
    }
    int32_t idx = pool_size++;
    memset(pool[idx].children, 0xFF, sizeof(pool[idx].children)); /* -1 in two's complement */
    pool[idx].token_id = -1;
    return idx;
}

static void insert(const char *key, int len, int id) {
    int32_t node = 0; /* root is always index 0 */
    for (int i = 0; i < len; i++) {
        unsigned char c = (unsigned char)key[i];
        if (pool[node].children[c] == -1) {
            pool[node].children[c] = alloc_node();
        }
        node = pool[node].children[c];
    }
    pool[node].token_id = id;
}

/* --- Vocab loading --- */

static uint32_t read_uint32_be(FILE *f) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
           ((uint32_t)buf[2] << 8)  |  (uint32_t)buf[3];
}

static uint16_t read_uint16_be(FILE *f) {
    unsigned char buf[2];
    if (fread(buf, 1, 2, f) != 2) return 0;
    return ((uint16_t)buf[0] << 8) | (uint16_t)buf[1];
}

static int load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("Error opening vocab file"); exit(1); }

    uint32_t count = read_uint32_be(f);
    printf("Loading %u tokens from vocab...\n", count);

    /* Init pool and allocate root */
    pool_cap = INITIAL_POOL_CAP;
    pool_size = 0;
    pool = malloc((size_t)pool_cap * sizeof(TrieNode));
    if (!pool) { fprintf(stderr, "Out of memory\n"); exit(1); }
    alloc_node(); /* root = index 0 */

    char token[MAX_TOKEN_LEN];
    for (uint32_t i = 0; i < count; i++) {
        uint32_t len = read_uint32_be(f);
        if (len >= MAX_TOKEN_LEN) {
            fprintf(stderr, "Token too long: %u\n", len);
            exit(1);
        }
        if (fread(token, 1, len, f) != len) {
            fprintf(stderr, "Error reading token %u\n", i);
            exit(1);
        }
        insert(token, (int)len, (int)i);
    }
    fclose(f);
    printf("Trie built: %d nodes (%.1f MB)\n", pool_size,
           (double)pool_size * sizeof(TrieNode) / (1024.0 * 1024.0));
    return (int)count;
}

/* --- Word-level template structures --- */

#define MAX_TMPL_WORDS 16
#define WT_HASH_BITS   16
#define WT_HASH_SIZE   (1 << WT_HASH_BITS)
#define WT_HASH_MASK   (WT_HASH_SIZE - 1)
#define MAX_TEMPLATES  8192
#define MAX_MATCHES    (2 * 1024 * 1024)

typedef struct {
    uint16_t vocab_id;
    uint8_t  num_words;
    uint8_t  slot_pos;      /* 0-indexed: which word is the slot */
    char    *words[MAX_TMPL_WORDS];   /* word strings (NULL at slot_pos) */
    int      word_lens[MAX_TMPL_WORDS];
} WordTemplate;

typedef struct {
    int32_t *indices;  /* indices into wt_templates[] */
    int32_t  count;
    int32_t  cap;
} WTBucket;

typedef struct {
    size_t boundary;     /* where trie encoding should stop before this match */
    size_t match_start;  /* first byte of the first template word */
    size_t match_end;    /* byte past the last template word */
    size_t fill_ws_start; /* start of fill's leading whitespace */
    size_t fill_end;     /* byte past the fill word */
    uint16_t template_id;
} TemplateMatch;

static WordTemplate wt_templates[MAX_TEMPLATES];
static int wt_num_templates = 0;
static uint32_t wt_base_id = 0;
static WTBucket wt_index[WT_HASH_SIZE];

static inline int is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/* FNV-1a hash for word bytes */
static uint32_t word_hash(const char *s, int len) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned char)s[i];
        h *= 16777619u;
    }
    return h & WT_HASH_MASK;
}

/* Compare by num_words descending for longest-first matching */
static int cmp_wt_desc(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return (int)wt_templates[ib].num_words - (int)wt_templates[ia].num_words;
}

static void load_word_templates(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("Error opening templates file"); exit(1); }

    wt_num_templates = (int)read_uint32_be(f);
    wt_base_id = read_uint32_be(f);

    if (wt_num_templates > MAX_TEMPLATES) {
        fprintf(stderr, "Too many templates: %d (max %d)\n", wt_num_templates, MAX_TEMPLATES);
        exit(1);
    }

    printf("Loading %d word-templates (base_id=%u)...\n", wt_num_templates, wt_base_id);

    for (int i = 0; i < wt_num_templates; i++) {
        uint8_t num_words, slot_pos;
        if (fread(&num_words, 1, 1, f) != 1) { fprintf(stderr, "Read error\n"); exit(1); }
        if (fread(&slot_pos, 1, 1, f) != 1) { fprintf(stderr, "Read error\n"); exit(1); }

        wt_templates[i].vocab_id = (uint16_t)(wt_base_id + i);
        wt_templates[i].num_words = num_words;
        wt_templates[i].slot_pos = slot_pos;

        for (int j = 0; j < num_words; j++) {
            if (j == slot_pos) {
                wt_templates[i].words[j] = NULL;
                wt_templates[i].word_lens[j] = 0;
                continue;
            }
            uint16_t wlen = read_uint16_be(f);
            char *w = malloc(wlen + 1);
            if (fread(w, 1, wlen, f) != wlen) { fprintf(stderr, "Read error\n"); exit(1); }
            w[wlen] = '\0';
            wt_templates[i].words[j] = w;
            wt_templates[i].word_lens[j] = wlen;
        }
    }
    fclose(f);

    /* Build index by first non-slot word (the word used for lookup).
     * For slot_pos==0, index by word[1] (first literal after slot).
     * For slot_pos>0, index by word[0]. */
    memset(wt_index, 0, sizeof(wt_index));

    for (int i = 0; i < wt_num_templates; i++) {
        int key_idx = (wt_templates[i].slot_pos == 0) ? 1 : 0;
        char *w = wt_templates[i].words[key_idx];
        int wlen = wt_templates[i].word_lens[key_idx];
        uint32_t h = word_hash(w, wlen);
        WTBucket *b = &wt_index[h];
        if (b->count >= b->cap) {
            b->cap = b->cap ? b->cap * 2 : 8;
            b->indices = realloc(b->indices, b->cap * sizeof(int32_t));
        }
        b->indices[b->count++] = i;
    }

    /* Sort each bucket by num_words descending */
    for (int i = 0; i < WT_HASH_SIZE; i++) {
        if (wt_index[i].count > 1)
            qsort(wt_index[i].indices, wt_index[i].count, sizeof(int32_t), cmp_wt_desc);
    }

    printf("Word-template index built.\n");
}

/* Check if a byte range encodes to exactly 1 trie token.
 * Returns the token ID, or -1 if not a single token. */
static int is_single_token(const unsigned char *data, int len) {
    int32_t node = 0;
    int last_id = -1;
    int last_len = 0;
    for (int i = 0; i < len; i++) {
        int32_t child = pool[node].children[data[i]];
        if (child == -1) break;
        node = child;
        if (pool[node].token_id != -1) {
            last_id = pool[node].token_id;
            last_len = i + 1;
        }
    }
    /* Must consume ALL bytes as exactly one token */
    return (last_len == len) ? last_id : -1;
}

/* Phase 1: Scan text for word-level template matches.
 * Returns number of matches found. */
static int scan_template_matches(const unsigned char *data, size_t len,
                                  TemplateMatch *matches) {
    int num_matches = 0;
    size_t pos = 0;

    while (pos < len && num_matches < MAX_MATCHES) {
        /* Skip whitespace */
        while (pos < len && is_ws(data[pos])) pos++;
        if (pos >= len) break;

        /* Find current word boundaries */
        size_t word_start = pos;
        while (pos < len && !is_ws(data[pos])) pos++;
        size_t word_end = pos;
        int word_len = (int)(word_end - word_start);

        /* --- Try templates where this word is word[0] (slot_pos > 0) --- */
        uint32_t h0 = word_hash((const char *)data + word_start, word_len);
        WTBucket *b0 = &wt_index[h0];
        for (int bi = 0; bi < b0->count; bi++) {
            int ti = b0->indices[bi];
            WordTemplate *t = &wt_templates[ti];
            if (t->slot_pos == 0) continue;  /* indexed by word[1], handled below */
            if (t->word_lens[0] != word_len) continue;
            if (memcmp(t->words[0], data + word_start, word_len) != 0) continue;

            /* word[0] matches. Walk forward checking remaining words. */
            size_t scan = word_end;
            int ok = 1;
            size_t fill_ws_start = 0, fill_end = 0;
            size_t last_end = word_end;

            for (int j = 1; j < t->num_words; j++) {
                /* Skip whitespace to next word */
                size_t ws_start = scan;
                while (scan < len && is_ws(data[scan])) scan++;
                if (scan >= len) { ok = 0; break; }

                size_t nw_start = scan;
                while (scan < len && !is_ws(data[scan])) scan++;
                size_t nw_end = scan;
                int nw_len = (int)(nw_end - nw_start);

                if (j == t->slot_pos) {
                    /* This is the fill slot — check ws+word is single token */
                    fill_ws_start = ws_start;
                    fill_end = nw_end;
                    int fill_len = (int)(nw_end - ws_start);
                    int tok = is_single_token(data + ws_start, fill_len);
                    if (tok < 0) { ok = 0; break; }
                } else {
                    /* Literal word — must match exactly */
                    if (nw_len != t->word_lens[j]) { ok = 0; break; }
                    if (memcmp(t->words[j], data + nw_start, nw_len) != 0) { ok = 0; break; }
                }
                last_end = nw_end;
            }

            if (ok) {
                /* boundary = match_start since slot_pos > 0 */
                matches[num_matches].boundary = word_start;
                matches[num_matches].match_start = word_start;
                matches[num_matches].match_end = last_end;
                matches[num_matches].fill_ws_start = fill_ws_start;
                matches[num_matches].fill_end = fill_end;
                matches[num_matches].template_id = t->vocab_id;
                num_matches++;
                /* Skip past the match: rewind pos to last_end so outer loop
                 * continues from there. We need to set pos to just past the
                 * match so the outer while loop doesn't re-process words
                 * inside the match. */
                pos = last_end;
                goto next_word;
            }
        }

        /* --- Try templates where slot_pos==0, indexed by word[1].
         * This word could be word[1] if the previous word was the fill slot. --- */
        /* For slot_pos==0 templates, we need to look backward to find the
         * fill word. But we don't track previous words. Instead, index these
         * templates by word[1] and when we see a matching word[1], look
         * backward for the fill. */
        {
            uint32_t h1 = word_hash((const char *)data + word_start, word_len);
            WTBucket *b1 = &wt_index[h1];
            for (int bi = 0; bi < b1->count; bi++) {
                int ti = b1->indices[bi];
                WordTemplate *t = &wt_templates[ti];
                if (t->slot_pos != 0) continue;  /* only slot_pos==0 here */
                /* word[1] should match current word */
                if (t->word_lens[1] != word_len) continue;
                if (memcmp(t->words[1], data + word_start, word_len) != 0) continue;

                /* Look backward for fill word (word[0] = slot).
                 * Find the whitespace + word preceding word_start. */
                if (word_start == 0) continue;
                size_t back = word_start;
                /* Skip whitespace backward */
                while (back > 0 && is_ws(data[back - 1])) back--;
                if (back == 0) continue;
                size_t fill_word_end = back;
                /* Find fill word start */
                while (back > 0 && !is_ws(data[back - 1])) back--;
                size_t fill_word_start = back;

                /* Find start of whitespace before fill word */
                size_t fill_ws = fill_word_start;
                while (fill_ws > 0 && is_ws(data[fill_ws - 1])) fill_ws--;
                /* fill_ws is now the byte after the previous non-ws char,
                 * or 0 if fill is at start of file.
                 * The fill's leading whitespace starts at fill_ws. */
                size_t fill_ws_start = fill_ws;

                /* But we only want the whitespace that directly precedes the
                 * fill word. Actually for encoding, the fill token is
                 * ws + fill_word, where ws is the whitespace between the
                 * previous word and the fill word. Since slot_pos==0, the fill
                 * is the first word, so ws_start is whatever is before it. */
                /* Actually: fill_ws_start should be the start of whitespace
                 * immediately before fill_word_start */
                fill_ws_start = fill_word_start;
                /* Back up over whitespace */
                while (fill_ws_start > 0 && is_ws(data[fill_ws_start - 1])) fill_ws_start--;

                /* Check ws+fill_word is a single token */
                int fill_len = (int)(fill_word_end - fill_ws_start);
                int tok = is_single_token(data + fill_ws_start, fill_len);
                if (tok < 0) continue;

                /* Check remaining words (word[2], word[3], ...) forward */
                size_t scan = word_end;
                int ok = 1;
                size_t last_end = word_end;
                for (int j = 2; j < t->num_words; j++) {
                    while (scan < len && is_ws(data[scan])) scan++;
                    if (scan >= len) { ok = 0; break; }
                    size_t nw_start = scan;
                    while (scan < len && !is_ws(data[scan])) scan++;
                    int nw_len = (int)(scan - nw_start);
                    if (nw_len != t->word_lens[j]) { ok = 0; break; }
                    if (memcmp(t->words[j], data + nw_start, nw_len) != 0) { ok = 0; break; }
                    last_end = scan;
                }
                if (!ok) continue;

                /* Conflict check: does this overlap a previous match? */
                if (num_matches > 0) {
                    TemplateMatch *prev = &matches[num_matches - 1];
                    if (fill_ws_start < prev->match_end) continue;
                }

                /* boundary = fill_ws_start (before fill's whitespace) */
                matches[num_matches].boundary = fill_ws_start;
                matches[num_matches].match_start = word_start;
                matches[num_matches].match_end = last_end;
                matches[num_matches].fill_ws_start = fill_ws_start;
                matches[num_matches].fill_end = fill_word_end;
                matches[num_matches].template_id = t->vocab_id;
                num_matches++;
                pos = last_end;
                goto next_word;
            }
        }

next_word:
        ; /* continue to next word */
    }

    return num_matches;
}

/* --- Encoding --- */

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <vocab_file> <input_file> <output_file> [templates_file]\n", argv[0]);
        return 1;
    }

    int vocab_size = load_vocab(argv[1]);
    (void)vocab_size;

    /* Optionally load word-level templates */
    int have_templates = 0;
    if (argc >= 5) {
        load_word_templates(argv[4]);
        have_templates = 1;
    }

    /* mmap input file */
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror("Error opening input"); exit(1); }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); exit(1); }
    size_t input_len = (size_t)st.st_size;

    const unsigned char *data = mmap(NULL, input_len, PROT_READ,
                                     MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise((void *)data, input_len, MADV_SEQUENTIAL);
    close(fd);

    printf("Encoding %s (%zu bytes) -> %s...\n", argv[2], input_len, argv[3]);

    /* Phase 1: Scan for template matches (if templates loaded) */
    TemplateMatch *matches = NULL;
    int num_matches = 0;
    if (have_templates) {
        matches = malloc(MAX_MATCHES * sizeof(TemplateMatch));
        if (!matches) { fprintf(stderr, "Out of memory\n"); exit(1); }
        num_matches = scan_template_matches(data, input_len, matches);
        printf("Template scan: %d matches found.\n", num_matches);
    }

    /* Open output */
    FILE *fout = fopen(argv[3], "wb");
    if (!fout) { perror("Error opening output"); exit(1); }

    /* Output buffer */
    uint16_t *out_buf = malloc(OUTPUT_BUF_TOKENS * sizeof(uint16_t));
    if (!out_buf) { fprintf(stderr, "Out of memory\n"); exit(1); }
    int out_count = 0;

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* Helper macro to emit a token */
    #define EMIT(tok_id) do { \
        out_buf[out_count++] = (uint16_t)(tok_id); \
        if (out_count == OUTPUT_BUF_TOKENS) { \
            fwrite(out_buf, sizeof(uint16_t), out_count, fout); \
            out_count = 0; \
        } \
        total_tokens++; \
    } while(0)

    size_t pos = 0;
    long long total_tokens = 0;
    int match_idx = 0;

    while (pos < input_len) {
        /* Check if we've hit a template match boundary */
        if (have_templates && match_idx < num_matches &&
            pos >= matches[match_idx].boundary) {

            TemplateMatch *m = &matches[match_idx];

            /* Emit template ID */
            EMIT(m->template_id);

            /* Emit fill token (ws + fill_word encoded as single trie token) */
            int fill_len = (int)(m->fill_end - m->fill_ws_start);
            int fill_tok = is_single_token(data + m->fill_ws_start, fill_len);
            EMIT(fill_tok);

            pos = m->match_end;
            match_idx++;
        } else {
            /* Normal trie encoding, bounded by next match boundary */
            size_t limit = input_len;
            if (have_templates && match_idx < num_matches)
                limit = matches[match_idx].boundary;

            /* Greedy longest-match trie walk */
            int32_t node = 0;
            int best_id = -1;
            size_t best_len = 0;
            size_t remaining = limit - pos;
            size_t max_walk = remaining < MAX_TOKEN_LEN ? remaining : MAX_TOKEN_LEN;

            for (size_t i = 0; i < max_walk; i++) {
                unsigned char c = data[pos + i];
                int32_t child = pool[node].children[c];
                if (child == -1) break;
                node = child;

                if (i + 1 < max_walk) {
                    int32_t spec = pool[node].children[data[pos + i + 1]];
                    if (spec != -1)
                        _mm_prefetch((const char *)&pool[spec], _MM_HINT_T0);
                }

                if (pool[node].token_id != -1) {
                    best_id = pool[node].token_id;
                    best_len = i + 1;
                }
            }

            if (best_id == -1) {
                best_id = 1; /* <unk> */
                best_len = 1;
            }

            EMIT(best_id);
            pos += best_len;
        }

        if ((total_tokens & 0xFFFFF) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
            double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                             (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
            double mb_s = (double)pos / (1024.0 * 1024.0) / elapsed;
            printf("\r  %lld M tokens | %.0f / %.0f MB | %.1f MB/s",
                   total_tokens / 1000000,
                   (double)pos / (1024.0 * 1024.0),
                   (double)input_len / (1024.0 * 1024.0),
                   mb_s);
            fflush(stdout);
        }
    }

    #undef EMIT

    /* Flush remaining output */
    if (out_count > 0) {
        fwrite(out_buf, sizeof(uint16_t), out_count, fout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                     (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
    double mb_s = (double)input_len / (1024.0 * 1024.0) / elapsed;

    printf("\nDone. Total tokens: %lld  |  %.2f sec  |  %.1f MB/s",
           total_tokens, elapsed, mb_s);
    if (have_templates)
        printf("  |  %d template matches", num_matches);
    printf("\n");

    fclose(fout);
    free(out_buf);
    free(matches);
    munmap((void *)data, input_len);
    free(pool);
    return 0;
}
