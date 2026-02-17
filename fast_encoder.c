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

#define MAX_TOKEN_LEN 1024
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

/* --- Encoding --- */

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <vocab_file> <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    load_vocab(argv[1]);

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

    /* Open output */
    FILE *fout = fopen(argv[3], "wb");
    if (!fout) { perror("Error opening output"); exit(1); }

    /* Output buffer */
    uint16_t *out_buf = malloc(OUTPUT_BUF_TOKENS * sizeof(uint16_t));
    if (!out_buf) { fprintf(stderr, "Out of memory\n"); exit(1); }
    int out_count = 0;

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    size_t pos = 0;
    long long total_tokens = 0;

    while (pos < input_len) {
        /* Greedy longest-match trie walk */
        int32_t node = 0; /* root */
        int best_id = -1;
        size_t best_len = 0;
        size_t remaining = input_len - pos;
        size_t max_walk = remaining < MAX_TOKEN_LEN ? remaining : MAX_TOKEN_LEN;

        for (size_t i = 0; i < max_walk; i++) {
            unsigned char c = data[pos + i];
            int32_t child = pool[node].children[c];
            if (child == -1) break;
            node = child;

            /* Prefetch next child speculatively */
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
            /* No match — should not happen if vocab covers all bytes */
            best_id = 1;
            best_len = 1;
        }

        /* Append to output buffer */
        out_buf[out_count++] = (uint16_t)best_id;
        if (out_count == OUTPUT_BUF_TOKENS) {
            fwrite(out_buf, sizeof(uint16_t), out_count, fout);
            out_count = 0;
        }

        pos += best_len;
        total_tokens++;

        if ((total_tokens & 0xFFFFF) == 0) { /* every ~1M tokens */
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

    /* Flush remaining output */
    if (out_count > 0) {
        fwrite(out_buf, sizeof(uint16_t), out_count, fout);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double elapsed = (t_now.tv_sec - t_start.tv_sec) +
                     (t_now.tv_nsec - t_start.tv_nsec) / 1e9;
    double mb_s = (double)input_len / (1024.0 * 1024.0) / elapsed;

    printf("\nDone. Total tokens: %lld  |  %.2f sec  |  %.1f MB/s\n",
           total_tokens, elapsed, mb_s);

    fclose(fout);
    free(out_buf);
    munmap((void *)data, input_len);
    free(pool);
    return 0;
}
