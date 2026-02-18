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

#define MAX_TOKEN_LEN 4096
#define MAX_NGRAM 7
#define NGRAM_BUF_SIZE 4096

/* Chained hash table with arena-allocated nodes.
 * Per-thread tables: 2^23 (8M buckets, 64MB each).
 * Global merge tables: 2^27 (128M buckets, 1GB each).
 */
#define THREAD_BITS 23
#define THREAD_SIZE (1U << THREAD_BITS)
#define THREAD_MASK (THREAD_SIZE - 1)

#define GLOBAL_BITS 27
#define GLOBAL_SIZE (1U << GLOBAL_BITS)
#define GLOBAL_MASK (GLOBAL_SIZE - 1)

/* Arena — per-thread */
#define ARENA_BLOCK (128 * 1024 * 1024)

typedef struct Arena { char *base; size_t used, cap; struct Arena *prev; } Arena;

static Arena *arena_new(void) {
    Arena *a = malloc(sizeof(Arena));
    a->base = malloc(ARENA_BLOCK);
    a->used = 0;
    a->cap = ARENA_BLOCK;
    a->prev = NULL;
    return a;
}

static inline void *arena_alloc(Arena **ap, size_t size) {
    size = (size + 7) & ~(size_t)7;
    Arena *a = *ap;
    if (__builtin_expect(a->used + size > a->cap, 0)) {
        Arena *n = malloc(sizeof(Arena));
        n->base = malloc(ARENA_BLOCK);
        n->used = 0;
        n->cap = ARENA_BLOCK;
        n->prev = a;
        *ap = n;
        a = n;
    }
    void *p = a->base + a->used;
    a->used += size;
    return p;
}

static void arena_free_all(Arena *a) {
    while (a) {
        Arena *prev = a->prev;
        free(a->base);
        free(a);
        a = prev;
    }
}

/* Node: hash + key inline via flexible array */
typedef struct Node {
    struct Node *next;
    uint32_t hash32;  /* upper 32 bits for fast reject */
    uint32_t count;
    uint16_t key_len;
    char key[];
} Node;

/* --- Hash table ops --- */

static inline void ht_add(Node **buckets, uint32_t mask, Arena **ap,
                           const char *key, int len, uint64_t hash) {
    uint32_t idx = (uint32_t)hash & mask;
    uint32_t tag = (uint32_t)(hash >> 32);
    Node *n = buckets[idx];
    while (n) {
        if (n->hash32 == tag && n->key_len == (uint16_t)len &&
            memcmp(n->key, key, len) == 0) {
            n->count++;
            return;
        }
        n = n->next;
    }
    Node *node = arena_alloc(ap, sizeof(Node) + len);
    node->hash32 = tag;
    node->count = 1;
    node->key_len = (uint16_t)len;
    memcpy(node->key, key, len);
    node->next = buckets[idx];
    buckets[idx] = node;
}

static inline void ht_merge_node(Node **buckets, uint32_t mask, Arena **ap,
                                  const char *key, int len, uint32_t tag, uint32_t count) {
    uint64_t h = XXH3_64bits(key, len);
    uint32_t idx = (uint32_t)h & mask;
    uint32_t gtag = (uint32_t)(h >> 32);

    Node *n = buckets[idx];
    while (n) {
        if (n->hash32 == gtag && n->key_len == (uint16_t)len &&
            memcmp(n->key, key, len) == 0) {
            n->count += count;
            return;
        }
        n = n->next;
    }
    Node *node = arena_alloc(ap, sizeof(Node) + len);
    node->hash32 = gtag;
    node->count = count;
    node->key_len = (uint16_t)len;
    memcpy(node->key, key, len);
    node->next = buckets[idx];
    buckets[idx] = node;
}

/* Pre-computed character type */
static uint8_t char_type[256];

static void init_char_type(void) {
    for (int i = 0; i < 256; i++) {
        unsigned char c = (unsigned char)i;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_')
            char_type[i] = 1;
        else if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                 c == '\f' || c == '\v')
            char_type[i] = 2;
        else
            char_type[i] = 3;
    }
}

/* --- Per-thread context --- */

typedef struct {
    Node **atoms, **phrases;
    Arena *arena;
    const unsigned char *data;
    size_t start, end, file_size;
    long long token_count;
} ThreadCtx;

static void *count_thread(void *arg) {
    ThreadCtx *ctx = (ThreadCtx *)arg;
    const unsigned char *data = ctx->data;
    size_t pos = ctx->start;
    size_t end = ctx->end;

    /* Align start to token boundary */
    if (pos > 0) {
        uint8_t prev_type = char_type[data[pos - 1]];
        while (pos < end && char_type[data[pos]] == prev_type)
            pos++;
    }

    /* Extend end to complete last token */
    size_t real_end = end;
    if (end < ctx->file_size) {
        uint8_t end_type = char_type[data[end - 1]];
        while (real_end < ctx->file_size && char_type[data[real_end]] == end_type)
            real_end++;
    }

    /* Sliding window of previous atoms */
    const char *win_ptr[MAX_NGRAM - 1];
    int win_len[MAX_NGRAM - 1];
    int win_depth = 0;
    char ngram_buf[NGRAM_BUF_SIZE];
    long long tokens = 0;

    while (pos < real_end) {
        uint8_t type = char_type[data[pos]];
        size_t tok_start = pos;
        pos++;
        while (pos < real_end && char_type[data[pos]] == type)
            pos++;

        int tok_len = (int)(pos - tok_start);
        if (tok_len >= MAX_TOKEN_LEN) tok_len = MAX_TOKEN_LEN - 1;
        const char *tok = (const char *)data + tok_start;

        uint64_t tok_hash = XXH3_64bits(tok, tok_len);
        ht_add(ctx->atoms, THREAD_MASK, &ctx->arena, tok, tok_len, tok_hash);
        tokens++;

        /* Build n-grams from 2 to min(MAX_NGRAM, win_depth+1) */
        int max_n = win_depth + 1;
        if (max_n > MAX_NGRAM - 1) max_n = MAX_NGRAM - 1;

        for (int n = 1; n <= max_n; n++) {
            /* n-gram = win[win_depth-n] .. win[win_depth-1] + tok
             * That's (n+1) atoms total, i.e. an (n+1)-gram */
            int total_len = tok_len;
            int ok = 1;
            /* First pass: compute total length */
            for (int j = 0; j < n; j++) {
                total_len += win_len[win_depth - n + j];
            }
            if (total_len >= NGRAM_BUF_SIZE) continue;

            /* Second pass: build the n-gram */
            int off = 0;
            for (int j = 0; j < n; j++) {
                int idx = win_depth - n + j;
                memcpy(ngram_buf + off, win_ptr[idx], win_len[idx]);
                off += win_len[idx];
            }
            memcpy(ngram_buf + off, tok, tok_len);
            off += tok_len;

            uint64_t ng_hash = XXH3_64bits(ngram_buf, off);
            ht_add(ctx->phrases, THREAD_MASK, &ctx->arena, ngram_buf, off, ng_hash);
        }

        /* Shift window */
        if (win_depth < MAX_NGRAM - 1) {
            win_ptr[win_depth] = tok;
            win_len[win_depth] = tok_len;
            win_depth++;
        } else {
            /* Slide: drop oldest, shift left */
            for (int i = 0; i < MAX_NGRAM - 2; i++) {
                win_ptr[i] = win_ptr[i + 1];
                win_len[i] = win_len[i + 1];
            }
            win_ptr[MAX_NGRAM - 2] = tok;
            win_len[MAX_NGRAM - 2] = tok_len;
        }
    }

    ctx->token_count = tokens;
    return NULL;
}

/* --- Merge + dump --- */

static void merge_table(Node **dst, uint32_t dst_mask, Arena **ap,
                         Node **src, uint32_t src_size) {
    for (uint32_t i = 0; i < src_size; i++) {
        Node *n = src[i];
        while (n) {
            ht_merge_node(dst, dst_mask, ap, n->key, n->key_len, n->hash32, n->count);
            n = n->next;
        }
    }
}

static void dump_table(Node **buckets, uint32_t size, FILE *out, int min_freq) {
    for (uint32_t i = 0; i < size; i++) {
        Node *n = buckets[i];
        while (n) {
            if ((int)n->count >= min_freq) {
                fprintf(out, "%u ", n->count);
                for (int j = 0; j < n->key_len; j++) {
                    char c = n->key[j];
                    if (c == '\n') fputs("\\n", out);
                    else if (c == '\r') fputs("\\r", out);
                    else if (c == '\\') fputs("\\\\", out);
                    else fputc(c, out);
                }
                fputc('\n', out);
            }
            n = n->next;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file> [-j threads]\n", argv[0]);
        return 1;
    }

    init_char_type();

    int nthreads = 0;  /* 0 = auto */
    const char *input_path = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            nthreads = atoi(argv[++i]);
        }
    }

    /* mmap input */
    int fd = open(input_path, O_RDONLY);
    if (fd < 0) { perror("Error opening file"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;

    if (nthreads <= 0) {
        nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (nthreads < 1) nthreads = 1;
        if (nthreads > 32) nthreads = 32;
    }
    const unsigned char *data = mmap(NULL, file_size, PROT_READ,
                                     MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((void *)data, file_size, MADV_SEQUENTIAL);
    close(fd);

    printf("Processing %s (%zu bytes) with %d threads\n", input_path, file_size, nthreads);
    printf("Per-thread: 2 x %u buckets (%zu MB each)\n",
           THREAD_SIZE, (size_t)THREAD_SIZE * 8 >> 20);

    /* Allocate per-thread state */
    ThreadCtx *ctxs = calloc(nthreads, sizeof(ThreadCtx));
    size_t chunk = file_size / nthreads;

    for (int i = 0; i < nthreads; i++) {
        ctxs[i].atoms   = calloc(THREAD_SIZE, sizeof(Node *));
        ctxs[i].phrases = calloc(THREAD_SIZE, sizeof(Node *));
        ctxs[i].arena   = arena_new();
        ctxs[i].data    = data;
        ctxs[i].file_size = file_size;
        ctxs[i].start   = i * chunk;
        ctxs[i].end     = (i == nthreads - 1) ? file_size : (i + 1) * chunk;
    }

    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    pthread_t *threads = malloc(nthreads * sizeof(pthread_t));
    for (int i = 0; i < nthreads; i++)
        pthread_create(&threads[i], NULL, count_thread, &ctxs[i]);
    for (int i = 0; i < nthreads; i++)
        pthread_join(threads[i], NULL);

    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double count_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

    long long total_tokens = 0;
    for (int i = 0; i < nthreads; i++)
        total_tokens += ctxs[i].token_count;
    printf("Count: %lld tokens | %.2fs | %.1f MB/s\n",
           total_tokens, count_s, (double)file_size / (1024.0 * 1024.0) / count_s);

    /* Merge into global tables (2^27 = 128M buckets) */
    printf("Merging into global tables (%u buckets, %zu MB each)...\n",
           GLOBAL_SIZE, (size_t)GLOBAL_SIZE * 8 >> 20);

    Node **g_atoms   = calloc(GLOBAL_SIZE, sizeof(Node *));
    Node **g_phrases = calloc(GLOBAL_SIZE, sizeof(Node *));
    Arena *g_arena   = arena_new();

    for (int i = 0; i < nthreads; i++) {
        merge_table(g_atoms,   GLOBAL_MASK, &g_arena, ctxs[i].atoms,   THREAD_SIZE);
        merge_table(g_phrases, GLOBAL_MASK, &g_arena, ctxs[i].phrases, THREAD_SIZE);
        free(ctxs[i].atoms); free(ctxs[i].phrases);
        arena_free_all(ctxs[i].arena);
    }

    struct timespec t2;
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double merge_s = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
    printf("Merge: %.2fs\n", merge_s);

    printf("Writing tokenizer/counts.txt...\n");
    FILE *out = fopen("tokenizer/counts.txt", "w");
    if (!out) { perror("fopen"); return 1; }
    char *wb = malloc(16 * 1024 * 1024);
    setvbuf(out, wb, _IOFBF, 16 * 1024 * 1024);

    fprintf(out, "ATOMS\n");
    dump_table(g_atoms, GLOBAL_SIZE, out, 1);
    fprintf(out, "PHRASES\n");
    dump_table(g_phrases, GLOBAL_SIZE, out, 100);
    fclose(out);

    struct timespec t3;
    clock_gettime(CLOCK_MONOTONIC, &t3);
    double total_s = (t3.tv_sec - t0.tv_sec) + (t3.tv_nsec - t0.tv_nsec) / 1e9;
    printf("Total: %.2fs (count %.2f + merge %.2f + write %.2f) | %.1f MB/s\n",
           total_s, count_s, merge_s,
           (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) / 1e9,
           (double)file_size / (1024.0 * 1024.0) / total_s);

    munmap((void *)data, file_size);
    free(g_atoms); free(g_phrases);
    arena_free_all(g_arena);
    free(wb); free(ctxs); free(threads);
    printf("Done.\n");
    return 0;
}
