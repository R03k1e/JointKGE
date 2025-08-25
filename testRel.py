import torch

# ======================= 数据读取 ==========================

def load_vector_file(file_path):
    embedding_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            vector = list(map(float, parts))
            embedding_list.append(torch.tensor(vector))
    return torch.stack(embedding_list)

def load_label_file(file_path):
    id2name = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            name = line.strip()
            id2name[idx] = name
    return id2name

# ======================= 关系预测 ==========================

def compounde_transform(h, r):
    # 假设关系向量包含三个部分：旋转、缩放、平移，每部分维度和实体相同
    dim = h.shape[0]
    if r.shape[0] != dim * 3:
        raise ValueError(f"关系向量维度不匹配：实体向量维度为 {dim}，但关系向量维度为 {r.shape[0]}。请检查关系向量文件。")

    rotation = r[:dim]
    scale = r[dim:2*dim]
    translation = r[2*dim:]

    # 示例 CompoundE 变换：旋转 + 缩放 + 平移
    h_trans = h * scale + rotation + translation
    return h_trans

def score_fn(h, r, t):
    h_trans = compounde_transform(h, r)
    return -torch.norm(h_trans - t, p=2)

def find_entity_id(name, id2name):
    for k, v in id2name.items():
        if v == name:
            return k
    raise ValueError(f"实体名称 '{name}' 未找到，请检查标签文件。")

# ======================= 主程序 ==========================

def main():
    # 文件路径
    entity_emb_path = 'ent.tsv'
    entity_label_path = 'ent_labels.tsv'
    relation_emb_path = 'rel.tsv'
    relation_label_path = 'rel_labels.tsv'

    # 加载数据
    entity_embeddings = load_vector_file(entity_emb_path)
    entity_id2name = load_label_file(entity_label_path)
    relation_embeddings = load_vector_file(relation_emb_path)
    relation_id2name = load_label_file(relation_label_path)

    # 输入实体名
    head_name = input("请输入头实体名称: ").strip()
    tail_name = input("请输入尾实体名称: ").strip()

    head_id = find_entity_id(head_name, entity_id2name)
    tail_id = find_entity_id(tail_name, entity_id2name)

    head_emb = entity_embeddings[head_id]
    tail_emb = entity_embeddings[tail_id]

    # 遍历所有关系，计算得分
    scores = []
    for rel_id in range(relation_embeddings.shape[0]):
        rel_emb = relation_embeddings[rel_id]
        score = score_fn(head_emb, rel_emb, tail_emb)
        scores.append((rel_id, score.item()))

    # 输出 Top-K
    top_k = 5
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print("\n预测的最相关关系:")
    for rank, (rel_id, sc) in enumerate(sorted_scores[:top_k]):
        print(f"Top-{rank+1}: 关系ID={rel_id}, 关系名={relation_id2name.get(rel_id, '未知')}, 得分={sc:.4f}")

if __name__ == '__main__':
    main()
