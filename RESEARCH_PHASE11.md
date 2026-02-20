The human brain doesn't build every sentence from scratch using grammar rules, but rather pulls "pre-fabricated chunks" from a massive mental library.


===
Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!  
He said, “Wow, that is a really amazing vase! Can I buy it?” 
The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!”
So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. 
And that's how Ben found an amazing vase in the store!
===

```prompt
You are an expert ML engineer. Implement a **Temporal Graph Transformer Encoder** from scratch in PyTorch for narrative event modeling.

## Task
Build a model that processes **temporal event graphs** like this example:

```
Nodes: E1, E2, E3, E4, E5, E6 (events), Ben, Shopkeeper, Friends, Store, Home, SpecialVase (entities)

Temporal chain: E1 → E2 → E3 → E4 → E5 → E6

Event arguments (at time t):
E1: AGENT=Ben, ACTION=go_to, DEST=Store
E2: AGENT=Ben, ACTION=see, PATIENT=SpecialVase  
E3: AGENT=Ben, ACTION=ask_to_buy, TARGET=Shopkeeper, PATIENT=SpecialVase
E4: AGENT=Shopkeeper, ACTION=permit, RECIPIENT=Ben
E5: AGENT=Ben, ACTION=take_home, PATIENT=SpecialVase, DEST=Home
E6: AGENT=Ben, ACTION=show, PATIENT=SpecialVase, RECIPIENT=Friends
```

## Input Format
Take a batch of temporal graphs as:
```python
data = {
    'node_features': torch.Tensor[B, N, D]  # B=batch, N=nodes, D=features
    'edge_index': torch.LongTensor[2, E]    # Graph edges (events ↔ entities)
    'edge_types': torch.LongTensor[E]       # Role types (AGENT, PATIENT, etc.)
    'temporal_order': torch.LongTensor[T]   # Event timestamps/sequence [E1, E2,...]
    'node_types': torch.LongTensor[N]       # EVENT vs ENTITY
}
```

## Architecture Requirements

### 1. **Temporal Positional Encoding**
- Encode event sequence position: `PE(t, 2i) = sin(t / 10000^(2i/d))`
- Add to event nodes only (entities are timeless within window)

### 2. **Multi-Relational Graph Attention** 
- Edge-type aware attention: `α_ij = softmax( (h_i W_Q)(h_j W_K)^T / √d_k * rel_emb[edge_type_ij] )`
- Message passing respects roles (AGENT→EVENT, EVENT→PATIENT flows differently)

### 3. **Temporal Transformer Blocks** (L=4 layers)
```
for each layer:
    1. Intra-timestep Graph Attention (spatial): events t ↔ entities at t
    2. Inter-timestep Attention (temporal): events across time with temporal PE  
    3. FFN per node type
    4. LayerNorm + residual
```

### 4. **Hierarchical Outputs**
- `event_embs[T, D]` - final event representations (for next-event prediction)
- `entity_embs[N_entities, D]` - contextualized entity states  
- `graph_emb[1, D]` - [CLS] token or mean-pool for whole narrative

## Key Components to Code

1. **RelTypeEmbedding**: Lookup for 20+ roles (AGENT, PATIENT, DEST, etc.)
2. **TemporalGraphAttention**: GraphAttn + temporal masking 
3. **EventEntityCrossAttention**: Events query entities at same timestep
4. **TemporalSelfAttention**: Causal attention across event sequence only

## Training Objective (example)
```
# Next event prediction
loss = CrossEntropy(event_embs[:-1], next_event_labels) +
       Contrastive(entity_embs[Ben], entity_embs[Shopkeeper])  # character tracking
```

## Constraints
- Efficient: O(T*N^2 + T^2) attention, T=20 events, N=15 nodes max
- PyTorch Geometric compatible edge_index format
- No external libraries beyond torch, torch_geometric, torch_nn

## Deliverables
1. Complete `TemporalGraphTransformerEncoder` class
2. Forward pass that handles the batch format above  
3. Example usage with Ben-vase story graph
4. 100-line core implementation max

Output clean, runnable PyTorch code with comments explaining the human-like situation model updates at each layer.
```