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

#define SLOT_SENTINEL 0xFFFF
#define MIN_L 4
#define MAX_L 12
#define MAX_TEMPLATES 1000000

/* Hash Table Config */
#define THREAD_BITS 22
#define THREAD_SIZE (1U << THREAD_BITS)
#define THREAD_MASK (THREAD_SIZE - 1)

#define GLOBAL_BITS 25
#define GLOBAL_SIZE (1U << GLOBAL_BITS)
#define GLOBAL_MASK (GLOBAL_SIZE - 1)

#define ARENA_BLOCK (128 * 1024 * 1024)

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

typedef struct Node {
    struct Node *next;
    uint32_t hash32;
    uint32_t count;
    uint8_t  n_len;
    uint8_t  slot_pos;
    uint16_t ids[];
} Node;

static inline void ht_add(Node **buckets, uint32_t mask, Arena **ap,
                           const uint16_t *ids, int n, int slot_pos, uint64_t hash) {
    uint32_t idx = (uint32_t)hash & mask;
    uint32_t tag = (uint32_t)(hash >> 32);
    size_t kb = n * sizeof(uint16_t);
    for (Node *nd = buckets[idx]; nd; nd = nd->next) {
        if (nd->hash32 == tag && nd->n_len == (uint8_t)n && nd->slot_pos == (uint8_t)slot_pos &&
            memcmp(nd->ids, ids, kb) == 0) {
            nd->count++;
            return;
        }
    }
    Node *node = arena_alloc(ap, sizeof(Node) + kb);
    node->hash32 = tag; node->count = 1; node->n_len = (uint8_t)n; node->slot_pos = (uint8_t)slot_pos;
    memcpy(node->ids, ids, kb);
    node->next = buckets[idx]; buckets[idx] = node;
}

static inline void ht_merge(Node **dst, uint32_t mask, Arena **ap, Node *src) {
    uint64_t h = XXH3_64bits(src->ids, src->n_len * sizeof(uint16_t)) ^ src->slot_pos;
    uint32_t idx = (uint32_t)h & mask;
    uint32_t tag = (uint32_t)(h >> 32);
    size_t kb = src->n_len * sizeof(uint16_t);
    for (Node *nd = dst[idx]; nd; nd = nd->next) {
        if (nd->hash32 == tag && nd->n_len == src->n_len && nd->slot_pos == src->slot_pos &&
            memcmp(nd->ids, src->ids, kb) == 0) {
            nd->count += src->count;
            return;
        }
    }
    Node *node = arena_alloc(ap, sizeof(Node) + kb);
    memcpy(node, src, sizeof(Node) + kb);
    node->next = dst[idx]; dst[idx] = node;
}

static uint16_t *g_tokens;
static size_t    g_n_tokens;

typedef struct {
    int tid, n_threads;
    Node **phrases;
    Arena *arena;
} ThreadCtx;

static void *mine_thread(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    size_t start = g_n_tokens * ctx->tid / ctx->n_threads;
    size_t end = g_n_tokens * (ctx->tid + 1) / ctx->n_threads;

    uint16_t frame[MAX_L];
    for (size_t i = start; i < end && i + MAX_L <= g_n_tokens; i++) {
        for (int L = MIN_L; L <= MAX_L && i + L <= g_n_tokens; L++) {
            memcpy(frame, g_tokens + i, L * sizeof(uint16_t));
            /* Try each slot position except 0 */
            for (int k = 1; k < L; k++) {
                uint16_t old = frame[k];
                frame[k] = SLOT_SENTINEL;
                uint64_t h = XXH3_64bits(frame, L * sizeof(uint16_t)) ^ k;
                ht_add(ctx->phrases, THREAD_MASK, &ctx->arena, frame, L, k, h);
                frame[k] = old;
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
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <tokens.bin> <out_templates.txt> <min_freq> [threads]\n", argv[0]);
        return 1;
    }
    int min_freq = atoi(argv[3]);
    int n_threads = (argc > 4) ? atoi(argv[4]) : 8;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st; fstat(fd, &st);
    g_n_tokens = st.st_size / 2;
    g_tokens = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (g_tokens == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    ThreadCtx *ctxs = calloc(n_threads, sizeof(ThreadCtx));
    for (int i = 0; i < n_threads; i++) {
        ctxs[i].tid = i; ctxs[i].n_threads = n_threads;
        ctxs[i].phrases = calloc(THREAD_SIZE, sizeof(Node *));
        ctxs[i].arena = arena_new();
    }

    printf("Mining token templates from %zu tokens...\n", g_n_tokens);
    pthread_t *tids = malloc(n_threads * sizeof(pthread_t));
    for (int i = 0; i < n_threads; i++) pthread_create(&tids[i], NULL, mine_thread, &ctxs[i]);
    for (int i = 0; i < n_threads; i++) pthread_join(tids[i], NULL);

    printf("Merging...\n");
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
    for (uint32_t i = 0; i < GLOBAL_SIZE; i++) {
        for (Node *nd = g_table[i]; nd; nd = nd->next) {
            if (nd->count >= (uint32_t)min_freq) {
                fprintf(out, "%u %u %u", nd->count, (uint32_t)nd->n_len, (uint32_t)nd->slot_pos);
                for (int j = 0; j < nd->n_len; j++) fprintf(out, " %u", nd->ids[j]);
                fprintf(out, "\n");
            }
        }
    }
    fclose(out);
    return 0;
}
