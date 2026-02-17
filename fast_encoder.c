#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAX_VOCAB_SIZE 70000 
#define BUFFER_SIZE 1024 * 1024 
#define MAX_TOKEN_LEN 1024

typedef struct TrieNode {
    struct TrieNode *children[256];
    int token_id; // -1 if not a token end
} TrieNode;

TrieNode *root;
char vocab[MAX_VOCAB_SIZE][256]; // Simple fixed size tokens
int vocab_size = 0;

TrieNode *create_node() {
    TrieNode *node = (TrieNode *)malloc(sizeof(TrieNode));
    if (node) {
        node->token_id = -1;
        for (int i = 0; i < 256; i++) node->children[i] = NULL;
    }
    return node;
}

void insert(const char *key, int len, int id) {
    TrieNode *node = root;
    for (int i = 0; i < len; i++) {
        unsigned char index = (unsigned char)key[i];
        if (!node->children[index]) {
            node->children[index] = create_node();
        }
        node = node->children[index];
    }
    node->token_id = id;
}

// Load vocab from file (length-prefixed format or newline separated?)
// Python greedyphrase.py uses struct.pack('>I', len) + bytes
// Let's implement reading that binary format.
// >I (4 bytes big endian) = count
// >I (4 bytes big endian) = len, then bytes...

uint32_t read_uint32_be(FILE *f) {
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return 0;
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

void load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("Error opening vocab file");
        exit(1);
    }

    uint32_t count = read_uint32_be(f);
    printf("Loading %u tokens from vocab...\n", count);
    
    root = create_node();
    vocab_size = count;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t len = read_uint32_be(f);
        char token[MAX_TOKEN_LEN]; // Assuming max token len fits in buffer
        if (len >= MAX_TOKEN_LEN) {
            fprintf(stderr, "Token too long: %u\n", len);
            exit(1);
        }
        if (fread(token, 1, len, f) != len) {
            fprintf(stderr, "Error reading token %u\n", i);
            exit(1);
        }
        token[len] = '\0'; // Null terminate for debug, but use len for insert
        insert(token, len, i);
    }
    fclose(f);
}

// Greedy match function
int match_longest(const unsigned char *buffer, int len, int *match_len) {
    TrieNode *node = root;
    int longest_id = -1;
    int longest_len = 0;
    
    for (int i = 0; i < len; i++) {
        unsigned char index = buffer[i];
        if (!node->children[index]) break;
        node = node->children[index];
        if (node->token_id != -1) {
            longest_id = node->token_id;
            longest_len = i + 1;
        }
    }
    
    *match_len = longest_len;
    return longest_id;
}

#define MAX_TOKEN_LEN 1024 // Max length of a single phrase token

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <vocab_file> <input_file> <output_file>\n", argv[0]);
        return 1;
    }

    const char *vocab_path = argv[1];
    const char *input_path = argv[2];
    const char *output_path = argv[3];

    load_vocab(vocab_path);
    
    FILE *fin = fopen(input_path, "rb");
    if (!fin) { perror("Error opening input"); exit(1); }
    
    FILE *fout = fopen(output_path, "wb");
    if (!fout) { perror("Error opening output"); exit(1); }
    
    printf("Encoding %s -> %s...\n", input_path, output_path);
    
    // We need a sliding window buffer because greedy matching requires lookahead.
    // However, our tokens are rarely huge. A circular buffer or just a large buffer with refill is fine.
    // Simple approach: Read large chunk, process until near end, memmove remaining, read more.
    
    unsigned char buffer[BUFFER_SIZE];
    size_t data_in_buffer = 0;
    size_t file_offset = 0;
    int eof = 0;
    
    long long total_tokens = 0;

    while (!eof || data_in_buffer > 0) {
        // Refill buffer if needed
        if (!eof && data_in_buffer < MAX_TOKEN_LEN) {
            size_t bytes_to_read = BUFFER_SIZE - data_in_buffer;
            size_t read = fread(buffer + data_in_buffer, 1, bytes_to_read, fin);
            data_in_buffer += read;
            if (read < bytes_to_read) eof = 1;
        }
        
        if (data_in_buffer == 0) break;

        // Greedy match at start of buffer
        int match_len = 0;
        int token_id = match_longest(buffer, (data_in_buffer > MAX_TOKEN_LEN ? MAX_TOKEN_LEN : data_in_buffer), &match_len);
        
        if (token_id == -1) {
            // No match? This shouldn't happen if vocab has bytes.
            // Fallback: take 1 byte as <unk> (ID 1 usually) or verify vocab.
            // Assuming ID 1 is <unk>.
            token_id = 1; 
            match_len = 1;
        }
        
        // Write token (uint16 big endian? Or just native for internal use? Python struct uses native usually, or we specify)
        // Let's use uint16 LE (standard) or BE. Let's match typical binary format.
        // I'll write uint16 Little Endian for simplicity with np.fromfile or torch.frombuffer
        uint16_t out_token = (uint16_t)token_id;
        fwrite(&out_token, sizeof(uint16_t), 1, fout);
        total_tokens++;
        
        // Shift buffer
        if (match_len < data_in_buffer) {
            memmove(buffer, buffer + match_len, data_in_buffer - match_len);
        }
        data_in_buffer -= match_len;
        
        if (total_tokens % 1000000 == 0) {
            printf("Encoded %lld M tokens...\r", total_tokens / 1000000);
            fflush(stdout);
        }
    }
    
    printf("\nDone. Total tokens: %lld\n", total_tokens);
    fclose(fin);
    fclose(fout);
    return 0;
}
