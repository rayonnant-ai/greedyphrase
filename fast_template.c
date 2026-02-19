/* fast_template.c — Apply template matching to a token stream.
 *
 * Greedy left-to-right template substitution: for each position,
 * try longest-match-first template. If all literal positions match,
 * emit template_id + fill tokens, skip the matched span.
 *
 * Usage: fast_template <templates_file> <input.tokens> <output.tokens> [meta.bin]
 *
 * templates_file: binary format from greedyphrase.py save()
 *   [uint32 BE: num_templates]
 *   [uint32 BE: base_id]
 *   per template: [uint8 length][uint8 num_slots][uint16 BE frame[length]]
 *
 * Token files: uint16 little-endian arrays.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

#define SLOT_SENTINEL 0xFFFFu
#define MAX_TEMPLATES 8192
#define MAX_FRAME_LEN 32
#define INDEX_SIZE    65536

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  | (uint32_t)b[3];
}

static inline uint16_t read_be16(FILE *f) {
    unsigned char b[2];
    fread(b, 1, 2, f);
    return ((uint16_t)b[0] << 8) | (uint16_t)b[1];
}

typedef struct {
    uint16_t vocab_id;
    uint8_t  length;
    uint8_t  num_slots;
    uint16_t frame[MAX_FRAME_LEN];
    uint8_t  slot_pos[MAX_FRAME_LEN];
    uint8_t  n_slots;
} Template;

/* Index bucket: template indices sorted by length descending */
typedef struct {
    int *idx;
    int  count;
    int  cap;
} Bucket;

static Template templates[MAX_TEMPLATES];
static int n_templates;
static Bucket index_table[INDEX_SIZE];

static void bucket_add(Bucket *b, int ti) {
    if (b->count >= b->cap) {
        b->cap = b->cap ? b->cap * 2 : 4;
        b->idx = realloc(b->idx, b->cap * sizeof(int));
    }
    b->idx[b->count++] = ti;
}

static int cmp_len_desc(const void *a, const void *b) {
    return (int)templates[*(const int *)b].length -
           (int)templates[*(const int *)a].length;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <templates> <in.tokens> <out.tokens> [meta.bin]\n", argv[0]);
        return 1;
    }
    double t0 = now_sec();

    /* ── Load templates ── */
    FILE *tf = fopen(argv[1], "rb");
    if (!tf) { perror(argv[1]); return 1; }

    n_templates = (int)read_be32(tf);
    uint32_t base_id = read_be32(tf);
    if (n_templates > MAX_TEMPLATES) {
        fprintf(stderr, "Too many templates (%d > %d)\n", n_templates, MAX_TEMPLATES);
        return 1;
    }

    for (int i = 0; i < n_templates; i++) {
        uint8_t length, num_slots;
        fread(&length, 1, 1, tf);
        fread(&num_slots, 1, 1, tf);
        templates[i].vocab_id = (uint16_t)(base_id + i);
        templates[i].length = length;
        templates[i].num_slots = num_slots;
        templates[i].n_slots = 0;
        for (int j = 0; j < length; j++) {
            templates[i].frame[j] = read_be16(tf);
            if (templates[i].frame[j] == SLOT_SENTINEL)
                templates[i].slot_pos[templates[i].n_slots++] = (uint8_t)j;
        }
    }
    fclose(tf);

    /* ── Build index: first literal token → template list ── */
    memset(index_table, 0, sizeof(index_table));
    for (int i = 0; i < n_templates; i++) {
        for (int j = 0; j < templates[i].length; j++) {
            if (templates[i].frame[j] != SLOT_SENTINEL) {
                bucket_add(&index_table[templates[i].frame[j]], i);
                break;
            }
        }
    }
    for (int i = 0; i < INDEX_SIZE; i++)
        if (index_table[i].count > 1)
            qsort(index_table[i].idx, index_table[i].count, sizeof(int), cmp_len_desc);

    fprintf(stderr, "  Loaded %d templates (base_id=%u, %.1fs)\n",
            n_templates, base_id, now_sec() - t0);

    /* ── mmap input tokens ── */
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) { perror(argv[2]); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t in_bytes = (size_t)st.st_size;
    uint32_t n_tok = (uint32_t)(in_bytes / 2);
    uint16_t *in = mmap(NULL, in_bytes, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (in == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(in, in_bytes, MADV_SEQUENTIAL);
    close(fd);

    /* ── Allocate output ── */
    uint16_t *out = malloc(in_bytes);
    uint8_t *meta = (argc > 4) ? malloc(n_tok) : NULL;
    uint32_t op = 0, matches = 0;

    /* ── Greedy template matching ── */
    double t1 = now_sec();
    uint32_t pos = 0;
    while (pos < n_tok) {
        int matched = 0;
        Bucket *b = &index_table[in[pos]];
        for (int bi = 0; bi < b->count; bi++) {
            Template *t = &templates[b->idx[bi]];
            int L = t->length;
            if (pos + (uint32_t)L > n_tok) continue;

            int ok = 1;
            for (int j = 0; j < L; j++) {
                if (t->frame[j] == SLOT_SENTINEL) continue;
                if (in[pos + j] != t->frame[j]) { ok = 0; break; }
            }
            if (ok) {
                out[op] = t->vocab_id;
                if (meta) meta[op] = 1;
                op++;
                for (int s = 0; s < t->n_slots; s++) {
                    out[op] = in[pos + t->slot_pos[s]];
                    if (meta) meta[op] = 2;
                    op++;
                }
                pos += (uint32_t)L;
                matched = 1;
                matches++;
                break;
            }
        }
        if (!matched) {
            out[op] = in[pos];
            if (meta) meta[op] = 0;
            op++;
            pos++;
        }
    }
    double elapsed = now_sec() - t1;

    fprintf(stderr, "  Templates matched %u times, tokens %u -> %u (saved %u, %.2fs)\n",
            matches, n_tok, op, n_tok - op, elapsed);

    /* ── Write output ── */
    FILE *of = fopen(argv[3], "wb");
    if (!of) { perror(argv[3]); return 1; }
    fwrite(out, sizeof(uint16_t), op, of);
    fclose(of);

    if (meta && argc > 4) {
        FILE *mf = fopen(argv[4], "wb");
        if (mf) { fwrite(meta, 1, op, mf); fclose(mf); }
    }

    munmap(in, in_bytes);
    free(out);
    if (meta) free(meta);
    for (int i = 0; i < INDEX_SIZE; i++)
        free(index_table[i].idx);

    fprintf(stderr, "  Total: %.1fs\n", now_sec() - t0);
    return 0;
}
