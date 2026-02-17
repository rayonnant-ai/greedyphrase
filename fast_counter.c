#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_TOKEN_LEN 256
#define HASH_SIZE 1000003 // Prime for hash table

typedef struct Node {
    char *key;
    int count;
    struct Node *next;
} Node;

Node *atom_table[HASH_SIZE];
Node *bigram_table[HASH_SIZE];
Node *trigram_table[HASH_SIZE];

unsigned long hash(char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    return hash % HASH_SIZE;
}

void add_count(Node **table, char *key) {
    unsigned long h = hash(key);
    Node *node = table[h];
    while (node) {
        if (strcmp(node->key, key) == 0) {
            node->count++;
            return;
        }
        node = node->next;
    }
    // New node
    Node *new_node = (Node *)malloc(sizeof(Node));
    new_node->key = strdup(key);
    new_node->count = 1;
    new_node->next = table[h];
    table[h] = new_node;
}

void dump_table(Node **table, FILE *out, int min_freq) {
    for (int i = 0; i < HASH_SIZE; i++) {
        Node *node = table[i];
        while (node) {
            if (node->count >= min_freq) {
                fprintf(out, "%d ", node->count);
                for (char *p = node->key; *p; p++) {
                    if (*p == '\n') fprintf(out, "\\n");
                    else if (*p == '\r') fprintf(out, "\\r");
                    else if (*p == '\\') fprintf(out, "\\\\");
                    else fputc(*p, out);
                }
                fputc('\n', out);
            }
            node = node->next;
        }
    }
}

int get_type(char c) {
    if (isalnum(c) || c == '_') return 1; // Word
    if (isspace(c)) return 2; // Space
    return 3; // Punctuation
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("Error opening file");
        return 1;
    }

    printf("Processing %s...\n", argv[1]);

    char buf[1024*1024]; // 1MB buffer
    char current_token[MAX_TOKEN_LEN] = {0};
    int current_len = 0;
    int current_type = 0;
    
    // Window for n-grams
    char w1[MAX_TOKEN_LEN] = {0}; // t-2
    char w2[MAX_TOKEN_LEN] = {0}; // t-1
    // current_token is t
    
    size_t n;
    long long total_tokens = 0;

    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++) {
            char c = buf[i];
            int type = get_type(c);

            if (current_len > 0 && type != current_type) {
                // Token boundary
                current_token[current_len] = '\0';
                
                // 1. Add Atom
                add_count(atom_table, current_token);
                total_tokens++;

                // 2. Add Bigram (w2 + current)
                if (w2[0] != 0) {
                    char bigram[MAX_TOKEN_LEN*2];
                    snprintf(bigram, sizeof(bigram), "%s%s", w2, current_token);
                    add_count(bigram_table, bigram);
                }

                // 3. Add Trigram (w1 + w2 + current)
                if (w1[0] != 0 && w2[0] != 0) {
                    char trigram[MAX_TOKEN_LEN*3];
                    snprintf(trigram, sizeof(trigram), "%s%s%s", w1, w2, current_token);
                    add_count(trigram_table, trigram);
                }

                // Shift window
                strcpy(w1, w2);
                strcpy(w2, current_token);
                
                // Reset for next token
                current_len = 0;
                if (total_tokens % 1000000 == 0) {
                    printf("Processed %lld M tokens...\r", total_tokens / 1000000);
                    fflush(stdout);
                }
            }

            if (current_len == 0) current_type = type;
            
            if (current_len < MAX_TOKEN_LEN - 1) {
                current_token[current_len++] = c;
            }
        }
    }
    
    // Process last token
    if (current_len > 0) {
        current_token[current_len] = '\0';
        add_count(atom_table, current_token);
    }

    printf("\nFinished reading. Writing to tokenizer/counts.txt...\n");
    FILE *out = fopen("tokenizer/counts.txt", "w");
    
    fprintf(out, "ATOMS\n");
    dump_table(atom_table, out, 1);
    
    fprintf(out, "BIGRAMS\n");
    dump_table(bigram_table, out, 50); // Filter infrequent phrases
    
    fprintf(out, "TRIGRAMS\n");
    dump_table(trigram_table, out, 50);
    
    fclose(out);
    fclose(f);
    
    printf("Done.\n");
    return 0;
}
