/*
 * phase10_miner.c — Greedy substring mining via suffix array
 *
 * Algorithm:
 *   1. mmap input file
 *   2. Build suffix array with SA-IS (O(N), byte alphabet)
 *   3. Build LCP array with Kasai's algorithm (O(N))
 *   4. Enumerate all repeated substrings via LCP interval tree walk
 *   5. Sort candidates by length DESC, then gain DESC
 *   6. Greedy replacement: longest-first, bitmap tracks consumed positions
 *   7. Write .vocab in big-endian length-prefixed format
 *
 * Compile: gcc -O3 -mavx2 -o phase10_miner phase10_miner.c -lm
 * Usage:   ./phase10_miner <input_file> [-o vocab_path] [-v vocab_size]
 *                          [-f min_freq] [-l max_len]
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

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ========================================================================
 * SA-IS: Suffix Array construction in O(N) for byte alphabet (sigma=256)
 * Based on Nong, Zhang, Chan (2009)
 * ======================================================================== */

/* Classify suffixes as S-type or L-type.  t[i]=1 means S-type. */
static void get_buckets(const int32_t *s, int32_t *bkt, int n, int sigma, int end) {
    memset(bkt, 0, (size_t)sigma * sizeof(int32_t));
    for (int i = 0; i < n; i++) bkt[s[i]]++;
    int sum = 0;
    if (end) {
        for (int i = 0; i < sigma; i++) { sum += bkt[i]; bkt[i] = sum; }
    } else {
        for (int i = 0; i < sigma; i++) { int t = bkt[i]; bkt[i] = sum; sum += t; }
    }
}

static void induce_sal(const int32_t *s, int32_t *sa, const uint8_t *t,
                       int32_t *bkt, int n, int sigma) {
    get_buckets(s, bkt, n, sigma, 0);
    for (int i = 0; i < n; i++) {
        int j = sa[i] - 1;
        if (sa[i] > 0 && !t[j])
            sa[bkt[s[j]]++] = j;
    }
}

static void induce_sas(const int32_t *s, int32_t *sa, const uint8_t *t,
                       int32_t *bkt, int n, int sigma) {
    get_buckets(s, bkt, n, sigma, 1);
    for (int i = n - 1; i >= 0; i--) {
        int j = sa[i] - 1;
        if (sa[i] > 0 && t[j])
            sa[--bkt[s[j]]] = j;
    }
}

static void sais_main(const int32_t *s, int32_t *sa, int n, int sigma) {
    uint8_t *t = calloc((size_t)n, 1);
    int32_t *bkt = malloc((size_t)sigma * sizeof(int32_t));

    /* Classify suffixes */
    t[n - 1] = 1;  /* sentinel is S-type */
    for (int i = n - 2; i >= 0; i--) {
        t[i] = (s[i] < s[i + 1] || (s[i] == s[i + 1] && t[i + 1])) ? 1 : 0;
    }

    /* Find LMS suffixes and place them at ends of their buckets */
    get_buckets(s, bkt, n, sigma, 1);
    memset(sa, -1, (size_t)n * sizeof(int32_t));
    for (int i = 1; i < n; i++) {
        if (t[i] && !t[i - 1])  /* LMS */
            sa[--bkt[s[i]]] = i;
    }

    /* Induce L-type and S-type */
    induce_sal(s, sa, t, bkt, n, sigma);
    induce_sas(s, sa, t, bkt, n, sigma);

    /* Compact sorted LMS substrings */
    int n1 = 0;
    for (int i = 0; i < n; i++) {
        if (sa[i] > 0 && t[sa[i]] && !t[sa[i] - 1])
            sa[n1++] = sa[i];
    }

    /* Name the LMS substrings */
    memset(sa + n1, -1, (size_t)(n - n1) * sizeof(int32_t));
    int name = 0, prev = -1;
    for (int i = 0; i < n1; i++) {
        int pos = sa[i];
        int diff = 0;
        if (prev == -1) {
            diff = 1;
        } else {
            /* Compare LMS substrings at prev and pos */
            for (int d = 0; ; d++) {
                int p1 = prev + d, p2 = pos + d;
                if (s[p1] != s[p2] || t[p1] != t[p2]) { diff = 1; break; }
                if (d > 0 && ((t[p1] && !t[p1 - 1]) || (t[p2] && !t[p2 - 1]))) {
                    /* Both reached LMS boundary — check if same */
                    if (s[p1] != s[p2] || t[p1] != t[p2]) diff = 1;
                    break;
                }
            }
        }
        if (diff) { name++; prev = pos; }
        /* Store name in sa[n1 + (pos/2)] area */
        sa[n1 + (pos >> 1)] = name - 1;
    }

    /* Collect reduced string (right-to-left to avoid overlap corruption) */
    int32_t *s1 = sa + n - n1;
    {
        int j = n1 - 1;
        for (int i = n - 1; i >= n1; i--) {
            if (sa[i] >= 0)
                s1[j--] = sa[i];
        }
    }

    /* Recurse if names are not unique */
    if (name < n1) {
        sais_main(s1, sa, n1, name);
    } else {
        for (int i = 0; i < n1; i++)
            sa[s1[i]] = i;
    }

    /* Rebuild LMS positions array */
    free(bkt);
    bkt = malloc((size_t)sigma * sizeof(int32_t));
    {
        int j = 0;
        /* Re-derive LMS positions */
        int32_t *lms_pos = malloc((size_t)n1 * sizeof(int32_t));
        for (int i = 1; i < n; i++) {
            if (t[i] && !t[i - 1])
                lms_pos[j++] = i;
        }

        /* Map sa[0..n1-1] from reduced indices to original positions */
        for (int i = 0; i < n1; i++)
            sa[i] = lms_pos[sa[i]];
        free(lms_pos);
    }

    /* Place sorted LMS suffixes at bucket ends */
    get_buckets(s, bkt, n, sigma, 1);
    memset(sa + n1, -1, (size_t)(n - n1) * sizeof(int32_t));
    for (int i = n1 - 1; i >= 0; i--) {
        int j = sa[i];
        sa[i] = -1;
        sa[--bkt[s[j]]] = j;
    }

    /* Final induction */
    induce_sal(s, sa, t, bkt, n, sigma);
    induce_sas(s, sa, t, bkt, n, sigma);

    free(t);
    free(bkt);
}

/* Build suffix array for byte input. Appends sentinel (0) internally.
 * Returns SA of length n (excluding sentinel). Caller frees SA. */
static int32_t *build_suffix_array(const uint8_t *data, int n) {
    int n1 = n + 1;  /* +1 for sentinel */
    int32_t *s = malloc((size_t)n1 * sizeof(int32_t));
    /* Extra padding for SA-IS naming step headroom (n1_lms + pos/2 can reach n1+1) */
    int32_t *sa = malloc(((size_t)n1 + 3) * sizeof(int32_t));

    for (int i = 0; i < n; i++) s[i] = (int32_t)data[i] + 1;  /* shift: 0 reserved for sentinel */
    s[n] = 0;  /* sentinel */

    sais_main(s, sa, n1, 258);  /* sigma = 256 bytes + 1 sentinel + 1 = 258 */
    free(s);

    /* Remove sentinel (sa[0] == n) and shift */
    int32_t *result = malloc((size_t)n * sizeof(int32_t));
    memcpy(result, sa + 1, (size_t)n * sizeof(int32_t));
    free(sa);
    return result;
}

/* ========================================================================
 * Kasai's LCP array construction — O(N)
 * ======================================================================== */

static int32_t *build_lcp(const uint8_t *data, const int32_t *sa, int n) {
    int32_t *rank = malloc((size_t)n * sizeof(int32_t));
    int32_t *lcp = malloc((size_t)n * sizeof(int32_t));

    for (int i = 0; i < n; i++) rank[sa[i]] = i;

    int k = 0;
    lcp[0] = 0;
    for (int i = 0; i < n; i++) {
        if (rank[i] == 0) { k = 0; continue; }
        int j = sa[rank[i] - 1];
        while (i + k < n && j + k < n && data[i + k] == data[j + k]) k++;
        lcp[rank[i]] = k;
        if (k > 0) k--;
    }

    free(rank);
    return lcp;
}

/* ========================================================================
 * LCP interval tree enumeration — single pass with stack
 *
 * Each LCP interval [lb, rb] with depth d represents a substring of
 * length d that occurs (rb - lb + 1) times.
 * ======================================================================== */

typedef struct {
    int32_t lb;      /* left bound in SA */
    int32_t rb;      /* right bound in SA (filled on pop) */
    int32_t depth;   /* LCP depth = substring length */
    int32_t freq;    /* rb - lb + 1 */
    int32_t excl_freq; /* freq - sum of direct child freqs (exclusive occurrences) */
} Candidate;

typedef struct {
    int32_t lb;
    int32_t depth;
    int32_t child_freq_sum; /* accumulated freq of child intervals */
} StackEntry;

static Candidate *enumerate_candidates(const int32_t *lcp, int n,
                                        int min_len, int min_freq, int max_len,
                                        int *out_count) {
    int cap = 1 << 20;  /* 1M initial */
    Candidate *cands = malloc((size_t)cap * sizeof(Candidate));
    int count = 0;

    int stack_cap = 1024;
    StackEntry *stack = malloc((size_t)stack_cap * sizeof(StackEntry));
    int sp = 0;

    /* Push initial entry */
    stack[sp].lb = 0;
    stack[sp].depth = 0;
    stack[sp].child_freq_sum = 0;
    sp++;

    for (int i = 1; i < n; i++) {
        int lb = i;
        while (sp > 0 && lcp[i] < stack[sp - 1].depth) {
            sp--;
            StackEntry top = stack[sp];
            int rb = i - 1;
            int freq = rb - top.lb + 1;
            int excl_freq = freq - top.child_freq_sum;
            lb = top.lb;

            /* Add this interval's freq to parent's child_freq_sum */
            if (sp > 0)
                stack[sp - 1].child_freq_sum += freq;

            if (top.depth >= min_len && top.depth <= max_len && freq >= min_freq) {
                if (count >= cap) {
                    cap *= 2;
                    cands = realloc(cands, (size_t)cap * sizeof(Candidate));
                }
                cands[count].lb = top.lb;
                cands[count].rb = rb;
                cands[count].depth = top.depth;
                cands[count].freq = freq;
                cands[count].excl_freq = excl_freq > 0 ? excl_freq : 0;
                count++;
            }
        }
        if (sp == 0 || lcp[i] > stack[sp - 1].depth) {
            if (sp >= stack_cap) {
                stack_cap *= 2;
                stack = realloc(stack, (size_t)stack_cap * sizeof(StackEntry));
            }
            stack[sp].lb = lb;
            stack[sp].depth = lcp[i];
            stack[sp].child_freq_sum = 0;
            sp++;
        }
    }

    /* Pop remaining */
    while (sp > 0) {
        sp--;
        StackEntry top = stack[sp];
        int rb = n - 1;
        int freq = rb - top.lb + 1;
        int excl_freq = freq - top.child_freq_sum;

        if (sp > 0)
            stack[sp - 1].child_freq_sum += freq;

        if (top.depth >= min_len && top.depth <= max_len && freq >= min_freq) {
            if (count >= cap) {
                cap *= 2;
                cands = realloc(cands, (size_t)cap * sizeof(Candidate));
            }
            cands[count].lb = top.lb;
            cands[count].rb = rb;
            cands[count].depth = top.depth;
            cands[count].freq = freq;
            cands[count].excl_freq = excl_freq > 0 ? excl_freq : 0;
            count++;
        }
    }

    free(stack);
    *out_count = count;
    return cands;
}

/* ========================================================================
 * AVX2 bitmap operations (with scalar fallback)
 * ======================================================================== */

/* Check if consumed[p..p+L-1] are all zero. Returns 1 if all clear. */
static inline int check_clear(const uint8_t *consumed, int64_t p, int L) {
#ifdef __AVX2__
    int64_t i = 0;
    /* Process 32 bytes at a time */
    for (; i + 32 <= L; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(consumed + p + i));
        if (!_mm256_testz_si256(v, v)) return 0;
    }
    /* Scalar tail */
    for (; i < L; i++) {
        if (consumed[p + i]) return 0;
    }
    return 1;
#else
    /* Scalar fallback: check 8 bytes at a time */
    int64_t i = 0;
    for (; i + 8 <= L; i += 8) {
        uint64_t v;
        memcpy(&v, consumed + p + i, 8);
        if (v) return 0;
    }
    for (; i < L; i++) {
        if (consumed[p + i]) return 0;
    }
    return 1;
#endif
}

/* Mark consumed[p..p+L-1] = 0xFF */
static inline void mark_consumed(uint8_t *consumed, int64_t p, int L) {
#ifdef __AVX2__
    int64_t i = 0;
    __m256i ones = _mm256_set1_epi8((char)0xFF);
    for (; i + 32 <= L; i += 32) {
        _mm256_storeu_si256((__m256i *)(consumed + p + i), ones);
    }
    for (; i < L; i++) {
        consumed[p + i] = 0xFF;
    }
#else
    memset(consumed + p, 0xFF, (size_t)L);
#endif
}

/* ========================================================================
 * Sorting: by length DESC, then by (len-1)*freq DESC within same length
 * ======================================================================== */

static int cmp_candidates(const void *a, const void *b) {
    const Candidate *ca = (const Candidate *)a;
    const Candidate *cb = (const Candidate *)b;
    /* Sort by gain = (length-1)*freq descending */
    int64_t ga = (int64_t)(ca->depth - 1) * ca->freq;
    int64_t gb = (int64_t)(cb->depth - 1) * cb->freq;
    if (ga > gb) return -1;
    if (ga < gb) return 1;
    /* Then by length descending as tiebreaker */
    return cb->depth - ca->depth;
}

/* ========================================================================
 * Position comparison for qsort
 * ======================================================================== */

static int cmp_int32(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

/* ========================================================================
 * Vocab output — big-endian length-prefixed binary
 * ======================================================================== */

static void write_uint32_be(FILE *f, uint32_t v) {
    uint8_t buf[4] = {
        (uint8_t)(v >> 24), (uint8_t)(v >> 16),
        (uint8_t)(v >> 8),  (uint8_t)v
    };
    fwrite(buf, 1, 4, f);
}

static void write_token(FILE *f, const uint8_t *data, int len) {
    write_uint32_be(f, (uint32_t)len);
    fwrite(data, 1, (size_t)len, f);
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(int argc, char *argv[]) {
    /* Defaults */
    const char *input_path = NULL;
    const char *vocab_path = "tokenizer/greedyphrase.vocab";
    int vocab_size = 65536;
    int min_freq = 50;
    int max_len = 200;

    /* Parse CLI */
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            input_path = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "-v") == 0 && i + 1 < argc) {
            vocab_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            min_freq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            max_len = atoi(argv[++i]);
        }
    }

    if (!input_path) {
        fprintf(stderr, "Usage: %s <input_file> [-o vocab_path] [-v vocab_size] "
                        "[-f min_freq] [-l max_len]\n", argv[0]);
        fprintf(stderr, "Defaults: vocab_size=%d, min_freq=%d, max_len=%d\n",
                65536, 50, 200);
        return 1;
    }

    int reserved = 260;  /* <pad>, <unk>, <s>, </s>, + 256 bytes */
    int phrase_budget = vocab_size - reserved;
    if (phrase_budget <= 0) {
        fprintf(stderr, "vocab_size must be > %d\n", reserved);
        return 1;
    }

    /* mmap input file */
    int fd = open(input_path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
    int64_t file_size = st.st_size;
    if (file_size > (int64_t)2000000000) {
        fprintf(stderr, "File too large (>2GB). SA uses int32_t.\n");
        return 1;
    }
    int n = (int)file_size;

    const uint8_t *data = mmap(NULL, (size_t)n, PROT_READ,
                                MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (data == MAP_FAILED) { perror("mmap"); return 1; }
    close(fd);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    printf("Phase 10: Greedy Substring Mining via Suffix Array\n");
    printf("Input: %s (%d bytes, %.1f MB)\n", input_path, n, n / 1e6);
    printf("Settings: vocab_size=%d, min_freq=%d, max_len=%d\n",
           vocab_size, min_freq, max_len);
    printf("Phrase budget: %d\n\n", phrase_budget);

    /* Step 1: Build suffix array */
    printf("Building suffix array (SA-IS)...\n");
    fflush(stdout);
    int32_t *sa = build_suffix_array(data, n);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  SA built in %.2fs\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
    fflush(stdout);

    /* Step 2: Build LCP array */
    printf("Building LCP array (Kasai)...\n");
    fflush(stdout);
    struct timespec t2;
    int32_t *lcp = build_lcp(data, sa, n);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    printf("  LCP built in %.2fs\n",
           (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9);
    fflush(stdout);

    /* Step 3: Enumerate candidates */
    printf("Enumerating candidates (LCP intervals)...\n");
    fflush(stdout);
    struct timespec t3;
    int num_cands = 0;
    Candidate *cands = enumerate_candidates(lcp, n, 2, min_freq, max_len, &num_cands);
    clock_gettime(CLOCK_MONOTONIC, &t3);
    printf("  %d candidates in %.2fs\n", num_cands,
           (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec) / 1e9);
    fflush(stdout);

    /* Free LCP — no longer needed */
    free(lcp);

    /* Step 4: Sort candidates */
    printf("Sorting candidates (length DESC, gain DESC)...\n");
    fflush(stdout);
    struct timespec t4;
    qsort(cands, (size_t)num_cands, sizeof(Candidate), cmp_candidates);
    clock_gettime(CLOCK_MONOTONIC, &t4);
    printf("  Sorted in %.2fs\n",
           (t4.tv_sec - t3.tv_sec) + (t4.tv_nsec - t3.tv_nsec) / 1e9);
    fflush(stdout);

    /* Step 5: Select top-K candidates by gain, deduplicate substrings */
    printf("Selecting top phrases...\n");
    fflush(stdout);

    typedef struct {
        int32_t sa_pos;   /* position in text for extracting bytes */
        int32_t length;
        int32_t freq;
    } Phrase;

    int phrase_cap = phrase_budget + 1024;
    Phrase *phrases = malloc((size_t)phrase_cap * sizeof(Phrase));
    int num_phrases = 0;

    struct timespec t5_start;
    clock_gettime(CLOCK_MONOTONIC, &t5_start);

    /* Already sorted by gain DESC. Just take top candidates,
     * skipping exact duplicates (same length+content at different SA intervals). */
    for (int ci = 0; ci < num_cands && num_phrases < phrase_budget; ci++) {
        Candidate *c = &cands[ci];
        phrases[num_phrases].sa_pos = sa[c->lb];
        phrases[num_phrases].length = c->depth;
        phrases[num_phrases].freq = c->freq;
        num_phrases++;
    }

    struct timespec t5_end;
    clock_gettime(CLOCK_MONOTONIC, &t5_end);
    printf("  Selected %d phrases in %.2fs\n", num_phrases,
           (t5_end.tv_sec - t5_start.tv_sec) + (t5_end.tv_nsec - t5_start.tv_nsec) / 1e9);
    fflush(stdout);

    free(cands);

    /* Step 6: Write vocab */
    printf("Writing vocab to %s...\n", vocab_path);
    fflush(stdout);
    FILE *fout = fopen(vocab_path, "wb");
    if (!fout) { perror("fopen vocab"); return 1; }

    int total_tokens = reserved + num_phrases;
    write_uint32_be(fout, (uint32_t)total_tokens);

    /* Special tokens */
    const char *specials[] = {"<pad>", "<unk>", "<s>", "</s>"};
    for (int i = 0; i < 4; i++) {
        write_token(fout, (const uint8_t *)specials[i], (int)strlen(specials[i]));
    }

    /* Byte tokens 0x00-0xFF */
    for (int i = 0; i < 256; i++) {
        uint8_t byte = (uint8_t)i;
        write_token(fout, &byte, 1);
    }

    /* Mined phrases in replacement order */
    for (int i = 0; i < num_phrases; i++) {
        write_token(fout, data + phrases[i].sa_pos, phrases[i].length);
    }

    fclose(fout);
    printf("  Wrote %d tokens (%d reserved + %d phrases)\n",
           total_tokens, reserved, num_phrases);

    /* Print top-10 phrases */
    printf("\nTop 10 phrases:\n");
    for (int i = 0; i < 10 && i < num_phrases; i++) {
        printf("  %3d: len=%3d freq=%7d  \"", i, phrases[i].length, phrases[i].freq);
        int show = phrases[i].length < 60 ? phrases[i].length : 60;
        for (int j = 0; j < show; j++) {
            uint8_t c = data[phrases[i].sa_pos + j];
            if (c == '\n') printf("\\n");
            else if (c == '\r') printf("\\r");
            else if (c == '\t') printf("\\t");
            else if (c >= 32 && c < 127) putchar(c);
            else printf("\\x%02x", c);
        }
        if (phrases[i].length > 60) printf("...");
        printf("\"\n");
    }

    /* Summary */
    struct timespec tend;
    clock_gettime(CLOCK_MONOTONIC, &tend);
    double total_s = (tend.tv_sec - t0.tv_sec) + (tend.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\nTotal time: %.2fs\n", total_s);

    free(phrases);
    free(sa);
    munmap((void *)data, (size_t)n);
    return 0;
}
