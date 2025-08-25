import torch

# ======================= Data Loading ==========================

def load_vector_file(file_path):
    """Load vector embeddings from a TSV file."""
    embedding_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            vector = list(map(float, parts))  # Convert string values to floats
            embedding_list.append(torch.tensor(vector))
    return torch.stack(embedding_list)  # Convert list of tensors to 2D tensor

def load_label_file(file_path):
    """Load ID-to-name mapping from a label file."""
    id2name = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            name = line.strip()
            id2name[idx] = name  # Map line number to entity/relation name
    return id2name

# ======================= Relation Prediction ==========================

def compounde_transform(h, r):
    """Apply CompoundE transformation: rotation + scaling + translation"""
    dim = h.shape[0]
    if r.shape[0] != dim * 3:
        raise ValueError(f"Relation vector dimension mismatch: Entity dim is {dim}, but relation dim is {r.shape[0]}. Check relation vector file.")

    # Split relation vector into three components
    rotation = r[:dim]
    scale = r[dim:2*dim]
    translation = r[2*dim:]

    # Apply transformation: h' = h * scale + rotation + translation
    h_trans = h * scale + rotation + translation
    return h_trans

def score_fn(h, r, t):
    """Calculate compatibility score between transformed head and tail"""
    h_trans = compounde_transform(h, r)
    return -torch.norm(h_trans - t, p=2)  # Negative L2 distance (higher score = better match)

def find_entity_id(name, id2name):
    """Find entity ID by name using reverse mapping"""
    for k, v in id2name.items():
        if v == name:
            return k
    raise ValueError(f"Entity name '{name}' not found. Check label file.")

# ======================= Main Program ==========================

def main():
    # File paths
    entity_emb_path = 'ent.tsv'
    entity_label_path = 'ent_labels.tsv'
    relation_emb_path = 'rel.tsv'
    relation_label_path = 'rel_labels.tsv'

    # Load embeddings and labels
    entity_embeddings = load_vector_file(entity_emb_path)
    entity_id2name = load_label_file(entity_label_path)
    relation_embeddings = load_vector_file(relation_emb_path)
    relation_id2name = load_label_file(relation_label_path)

    # Get user input
    head_name = input("Enter head entity name: ").strip()
    tail_name = input("Enter tail entity name: ").strip()

    # Convert names to IDs and get corresponding embeddings
    head_id = find_entity_id(head_name, entity_id2name)
    tail_id = find_entity_id(tail_name, entity_id2name)
    head_emb = entity_embeddings[head_id]
    tail_emb = entity_embeddings[tail_id]

    # Calculate scores for all possible relations
    scores = []
    for rel_id in range(relation_embeddings.shape[0]):
        rel_emb = relation_embeddings[rel_id]
        score = score_fn(head_emb, rel_emb, tail_emb)
        scores.append((rel_id, score.item()))  # Store (relation_id, score) tuple

    # Output top-K results
    top_k = 5
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)  # Sort by score descending
    print("\nPredicted most relevant relations:")
    for rank, (rel_id, sc) in enumerate(sorted_scores[:top_k]):
        print(f"Top-{rank+1}: RelationID={rel_id}, Name={relation_id2name.get(rel_id, 'Unknown')}, Score={sc:.4f}")

if __name__ == '__main__':
    main()
