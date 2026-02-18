#define XXH_INLINE_ALL
#include "xxhash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

#define SENTINEL_ID 0xFFFFFFFFu
#define LINE_SEP    0xFFFFFFFEu
#define MIN_WINDOW      5
#define MAX_WINDOW      15
#define MIN_LITERAL_LEN 15

/* Hash Table Config */
#define THREAD_BITS 22
#define THREAD_SIZE (1U << THREAD_BITS)
#define THREAD_MASK (THREAD_SIZE - 1)

#define GLOBAL_BITS 25
#define GLOBAL_SIZE (1U << GLOBAL_BITS)
#define GLOBAL_MASK (GLOBAL_SIZE - 1)

#define ARENA_BLOCK (128 * 1024 * 1024)

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

/* ─── Globals ─── */
static uint32_t *g_words;
static uint32_t  g_n_words;
static uint32_t *g_vocab_lens;
static char    **g_vocab_words;
static uint32_t  g_vocab_size;

/* ─── Threading ─── */
typedef struct {
    int tid, n_threads;
    Node **phrases;
    Arena *arena;
} ThreadCtx;

static void *mine_thread(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    uint32_t start = (uint32_t)((uint64_t)g_n_words * ctx->tid / ctx->n_threads);
    uint32_t end   = (uint32_t)((uint64_t)g_n_words * (ctx->tid + 1) / ctx->n_threads);

    /* Align to line boundary */
    if (start > 0) {
        while (start < g_n_words && g_words[start - 1] != LINE_SEP) start++;
    }

    uint32_t pos = start;
    while (pos < end && pos < g_n_words) {
        uint32_t lstart = pos;
        while (pos < g_n_words && g_words[pos] != LINE_SEP) pos++;
        uint32_t lend = pos;
        if (pos < g_n_words) pos++;

        uint32_t llen = lend - lstart;
        if (llen < MIN_WINDOW) continue;

        const uint32_t *line = g_words + lstart;

        for (uint32_t i = 0; i < llen; i++) {
            int total_lit = 0;
            for (int L = 1; L <= MAX_WINDOW && i + L <= llen; L++) {
                uint32_t wid = line[i + L - 1];
                if (wid != SENTINEL_ID) {
                    total_lit += g_vocab_lens[wid];
                }
                if (L >= MIN_WINDOW && total_lit >= MIN_LITERAL_LEN) {
                    /* Only keep if it has at least one sentinel */
                    int has_sentinel = 0;
                    for (int k = 0; k < L; k++) if (line[i + k] == SENTINEL_ID) { has_sentinel = 1; break; }
                    
                    if (has_sentinel) {
                        uint64_t h = XXH3_64bits(line + i, L * sizeof(uint32_t));
                        ht_add(ctx->phrases, THREAD_MASK, &ctx->arena, line + i, L, h);
                    }
                }
            }
        }
    }
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
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <masked_words.bin> <mask_vocab.bin> <out_templates.txt> <min_freq> [threads]\n", argv[0]);
        return 1;
    }
    int min_freq = atoi(argv[4]);
    int n_threads = (argc > 5) ? atoi(argv[5]) : (int)sysconf(_SC_NPROCESSORS_ONLN);

    /* Load vocab */
    FILE *vf = fopen(argv[2], "rb");
    if (!vf) { perror(argv[2]); return 1; }
    fread(&g_vocab_size, 4, 1, vf);
    g_vocab_lens = malloc(g_vocab_size * sizeof(uint32_t));
    g_vocab_words = malloc(g_vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < g_vocab_size; i++) {
        fread(&g_vocab_lens[i], 4, 1, vf);
        g_vocab_words[i] = malloc(g_vocab_lens[i] + 1);
        fread(g_vocab_words[i], 1, g_vocab_lens[i], vf);
        g_vocab_words[i][g_vocab_lens[i]] = '\0';
    }
    fclose(vf);

    /* mmap words */
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st;
    fstat(fd, &st);
    g_n_words = st.st_size / sizeof(uint32_t);
    g_words = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (g_words == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    ThreadCtx *ctxs = calloc(n_threads, sizeof(ThreadCtx));
    for (int i = 0; i < n_threads; i++) {
        ctxs[i].tid = i; ctxs[i].n_threads = n_threads;
        ctxs[i].phrases = calloc(THREAD_SIZE, sizeof(Node *));
        ctxs[i].arena = arena_new();
    }

    printf("Mining templates from %u words with %d threads...\n", g_n_words, n_threads);
    pthread_t *tids = malloc(n_threads * sizeof(pthread_t));
    for (int i = 0; i < n_threads; i++) pthread_create(&tids[i], NULL, mine_thread, &ctxs[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);

    printf("Merging results...\n");
    Node **g_table = calloc(GLOBAL_SIZE, sizeof(Node *));
    MergeCtx *mctxs = calloc(n_threads, sizeof(MergeCtx));
    for (int i = 0; i < n_threads; i++) {
        mctxs[i].shard_id = i; mctxs[i].n_shards = n_threads;
        mctxs[i].n_threads = n_threads; mctxs[i].ctxs = ctxs;
        mctxs[i].g_table = g_table; mctxs[i].g_arena = arena_new();
    }
    for (int i = 0; i < n_threads; i++) pthread_create(&tids[i], NULL, merge_thread, &mctxs[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);

    FILE *out = fopen(argv[3], "w");
    if (!out) { perror(argv[3]); return 1; }
    for (uint32_t i = 0; i < GLOBAL_SIZE; i++) {
        for (Node *nd = g_table[i]; nd; nd = nd->next) {
            if (nd->count >= (uint32_t)min_freq) {
                fprintf(out, "%u ", nd->count);
                for (int j = 0; j < nd->n_len; j++) {
                    if (nd->ids[j] == SENTINEL_ID) {
                        fprintf(out, ";?");
                    } else {
                        uint32_t wid = nd->ids[j];
                        fwrite(g_vocab_words[wid], 1, g_vocab_lens[wid], out);
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
