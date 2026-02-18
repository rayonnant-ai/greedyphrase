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
#include <pthread.h>

/* Constants */
#define MAX_WORD_LEN    256
#define MIN_WINDOW      5
#define MAX_WINDOW      15
#define MIN_LITERAL_LEN 15
#define SENTINEL_ID     0xFFFFFFFFu
#define LINE_SEP        0xFFFFFFFEu

/* Hash Table Config */
#define THREAD_BITS 20
#define THREAD_SIZE (1U << THREAD_BITS)
#define THREAD_MASK (THREAD_SIZE - 1)

#define GLOBAL_BITS 23
#define GLOBAL_SIZE (1U << GLOBAL_BITS)
#define GLOBAL_MASK (GLOBAL_SIZE - 1)

#define ARENA_BLOCK (64 * 1024 * 1024)

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

/* ─── Hash Table Node ─── */
typedef struct Node {
    struct Node *next;
    uint32_t hash32;
    uint32_t count;
    uint16_t n_len;
    uint32_t ids[];
} Node;

static inline void ht_add(Node **buckets, uint32_t mask, Arena **ap,
                           const uint32_t *ids, int n, uint64_t hash) {
    uint32_t idx = (uint32_t)hash & mask;
    uint32_t tag = (uint32_t)(hash >> 32);
    size_t kb = n * sizeof(uint32_t);
    for (Node *nd = buckets[idx]; nd; nd = nd->next) {
        if (nd->hash32 == tag && nd->n_len == (uint16_t)n &&
            memcmp(nd->ids, ids, kb) == 0) {
            nd->count++;
            return;
        }
    }
    Node *node = arena_alloc(ap, sizeof(Node) + kb);
    node->hash32 = tag; node->count = 1; node->n_len = (uint16_t)n;
    memcpy(node->ids, ids, kb);
    node->next = buckets[idx]; buckets[idx] = node;
}

static inline void ht_merge(Node **dst, uint32_t mask, Arena **ap, Node *src) {
    uint64_t h = XXH3_64bits(src->ids, src->n_len * sizeof(uint32_t));
    uint32_t idx = (uint32_t)h & mask;
    uint32_t tag = (uint32_t)(h >> 32);
    size_t kb = src->n_len * sizeof(uint32_t);
    for (Node *nd = dst[idx]; nd; nd = nd->next) {
        if (nd->hash32 == tag && nd->n_len == src->n_len &&
            memcmp(nd->ids, src->ids, kb) == 0) {
            nd->count += src->count;
            return;
        }
    }
    Node *node = arena_alloc(ap, sizeof(Node) + kb);
    memcpy(node, src, sizeof(Node) + kb);
    node->next = dst[idx]; dst[idx] = node;
}

/* ─── Linguistic Classification ─── */
static const char *MONTHS[] = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    "January", "February", "March", "April", "June", "July", "August", "September", "October", "November", "December"
};

static int is_month(const char *s, int len) {
    for (int i = 0; i < 23; i++) {
        if (len == (int)strlen(MONTHS[i]) && memcmp(s, MONTHS[i], len) == 0) return 1;
    }
    return 0;
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

static int is_proper_noun(const char *s, int len, int is_start_of_sentence) {
    if (len <= 2 || !isupper((unsigned char)s[0])) return 0;
    if (is_start_of_sentence) return 0;
    for (int i = 1; i < len; i++) {
        if (!islower((unsigned char)s[i])) return 0;
    }
    return 1;
}

/* ─── Word → ID mapping ─── */
typedef struct WNode { struct WNode *next; uint32_t tag, id; uint16_t len; char s[]; } WNode;
static WNode **wid_map;
static Arena *wid_arena;
static uint32_t next_wid = 0;
static char **v_words;
static int *v_lens;
static uint32_t v_cap = 0;
static pthread_mutex_t wid_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint32_t get_wid(const char *s, int len) {
    uint64_t h = XXH3_64bits(s, len);
    uint32_t idx = (uint32_t)h & 0x1FFFFFu;
    uint32_t tag = (uint32_t)(h >> 32);
    
    pthread_mutex_lock(&wid_mutex);
    for (WNode *nd = wid_map[idx]; nd; nd = nd->next) {
        if (nd->tag == tag && nd->len == len && memcmp(nd->s, s, len) == 0) {
            uint32_t id = nd->id;
            pthread_mutex_unlock(&wid_mutex);
            return id;
        }
    }
    uint32_t id = next_wid++;
    WNode *nd = arena_alloc(&wid_arena, sizeof(WNode) + len);
    nd->tag = tag; nd->id = id; nd->len = (uint16_t)len;
    memcpy(nd->s, s, len);
    nd->next = wid_map[idx]; wid_map[idx] = nd;
    
    if (id >= v_cap) {
        v_cap = v_cap ? v_cap * 2 : 1024 * 1024;
        v_words = realloc(v_words, v_cap * sizeof(char *));
        v_lens = realloc(v_lens, v_cap * sizeof(int));
    }
    char *cp = arena_alloc(&wid_arena, len);
    memcpy(cp, s, len);
    v_words[id] = cp; v_lens[id] = len;
    pthread_mutex_unlock(&wid_mutex);
    return id;
}

/* ─── Threading ─── */
typedef struct {
    int tid, n_threads;
    const char *text;
    size_t fsize;
    Node **phrases;
    Arena *arena;
} ThreadCtx;

static void *mine_thread(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    size_t start = ctx->fsize * ctx->tid / ctx->n_threads;
    size_t end = ctx->fsize * (ctx->tid + 1) / ctx->n_threads;

    if (start > 0) {
        while (start < ctx->fsize && ctx->text[start - 1] != '\n') start++;
    }
    if (end < ctx->fsize) {
        while (end < ctx->fsize && ctx->text[end - 1] != '\n') end++;
    }

    uint32_t *line_ids = malloc(8192 * sizeof(uint32_t));
    int *line_lit_lens = malloc(8192 * sizeof(int));
    
    size_t pos = start;
    while (pos < end) {
        int n_words = 0;
        int is_start = 1;
        while (pos < ctx->fsize && ctx->text[pos] != '\n') {
            while (pos < ctx->fsize && isspace((unsigned char)ctx->text[pos]) && ctx->text[pos] != '\n') pos++;
            if (pos >= ctx->fsize || ctx->text[pos] == '\n') break;
            
            size_t wstart = pos;
            while (pos < ctx->fsize && !isspace((unsigned char)ctx->text[pos])) pos++;
            int wlen = (int)(pos - wstart);
            
            uint32_t wid;
            int lit_len = 0;
            if (is_number(ctx->text + wstart, wlen) ||
                is_month(ctx->text + wstart, wlen) ||
                is_proper_noun(ctx->text + wstart, wlen, is_start)) {
                wid = SENTINEL_ID;
            } else {
                wid = get_wid(ctx->text + wstart, wlen);
                lit_len = wlen;
            }
            
            if (n_words < 8192) {
                line_ids[n_words] = wid;
                line_lit_lens[n_words] = lit_len;
                n_words++;
            }
            
            char last = ctx->text[pos - 1];
            if (last == '.' || last == '!' || last == '?') is_start = 1;
            else is_start = 0;
        }
        if (pos < ctx->fsize && ctx->text[pos] == '\n') pos++;

        for (int i = 0; i < n_words; i++) {
            int total_lit = 0;
            for (int L = 1; L <= MAX_WINDOW && i + L <= n_words; L++) {
                total_lit += line_lit_lens[i + L - 1];
                if (L >= MIN_WINDOW && total_lit >= MIN_LITERAL_LEN) {
                    uint64_t h = XXH3_64bits(line_ids + i, L * sizeof(uint32_t));
                    ht_add(ctx->phrases, THREAD_MASK, &ctx->arena, line_ids + i, L, h);
                }
            }
        }
    }

    free(line_ids);
    free(line_lit_lens);
    return NULL;
}

typedef struct {
    int shard_id, n_shards, n_threads;
    ThreadCtx *ctxs;
    Node **g_table;
    Arena *g_arena;
} MergeCtx;

static void *merge_thread(void *arg) {
    MergeCtx *ctx = (MergeCtx *)arg;
    uint32_t bs = (uint32_t)((uint64_t)THREAD_SIZE * ctx->shard_id / ctx->n_shards);
    uint32_t be = (uint32_t)((uint64_t)THREAD_SIZE * (ctx->shard_id + 1) / ctx->n_shards);

    for (int t = 0; t < ctx->n_threads; t++) {
        for (uint32_t b = bs; b < be; b++) {
            for (Node *nd = ctx->ctxs[t].phrases[b]; nd; nd = nd->next) {
                ht_merge(ctx->g_table, GLOBAL_MASK, &ctx->g_arena, nd);
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <text_file> <out_templates.txt> [min_freq] [threads]\n", argv[0]);
        return 1;
    }
    int min_freq = (argc > 3) ? atoi(argv[3]) : 10;
    int n_threads = (argc > 4) ? atoi(argv[4]) : (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (n_threads > 32) n_threads = 32;
    if (n_threads < 1) n_threads = 1;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st; fstat(fd, &st);
    const char *text = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (text == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    wid_map = calloc(0x200000, sizeof(WNode *));
    wid_arena = arena_new();

    ThreadCtx *ctxs = calloc(n_threads, sizeof(ThreadCtx));
    for (int i = 0; i < n_threads; i++) {
        ctxs[i].tid = i; ctxs[i].n_threads = n_threads;
        ctxs[i].text = text; ctxs[i].fsize = st.st_size;
        ctxs[i].phrases = calloc(THREAD_SIZE, sizeof(Node *));
        ctxs[i].arena = arena_new();
    }

    printf("Mining templates from %zu bytes with %d threads...\n", st.st_size, n_threads);
    pthread_t *tids = malloc(n_threads * sizeof(pthread_t));
    for (int i = 0; i < n_threads; i++) pthread_create(&tids[i], NULL, mine_thread, &ctxs[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);

    printf("Merging results into global table...\n");
    Node **g_table = calloc(GLOBAL_SIZE, sizeof(Node *));
    MergeCtx *mctxs = calloc(n_threads, sizeof(MergeCtx));
    for (int i = 0; i < n_threads; i++) {
        mctxs[i].shard_id = i; mctxs[i].n_shards = n_threads;
        mctxs[i].n_threads = n_threads; mctxs[i].ctxs = ctxs;
        mctxs[i].g_table = g_table; mctxs[i].g_arena = arena_new();
    }
    for (int i = 0; i < n_threads; i++) pthread_create(&tids[i], NULL, merge_thread, &mctxs[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);

    FILE *out = fopen(argv[2], "w");
    if (!out) { perror(argv[2]); return 1; }
    for (uint32_t i = 0; i < GLOBAL_SIZE; i++) {
        for (Node *nd = g_table[i]; nd; nd = nd->next) {
            if (nd->count >= (uint32_t)min_freq) {
                int has_sentinel = 0;
                for (int j = 0; j < nd->n_len; j++) if (nd->ids[j] == SENTINEL_ID) has_sentinel = 1;
                if (!has_sentinel) continue;

                fprintf(out, "%u ", nd->count);
                for (int j = 0; j < nd->n_len; j++) {
                    if (nd->ids[j] == SENTINEL_ID) {
                        fprintf(out, ";?");
                    } else {
                        uint32_t wid = nd->ids[j];
                        fwrite(v_words[wid], 1, v_lens[wid], out);
                    }
                    if (j < nd->n_len - 1) fprintf(out, " ");
                }
                fprintf(out, "\n");
            }
        }
    }
    fclose(out);

    return 0;
}