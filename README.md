
1. Overview
JT-KGE is designed for jointly learning:
Instance-level embeddings: Capturing relations between entities.
Type-level embeddings: Capturing hierarchical types of entities.
Entity-type associations: Linking entities to their semantic types.
The framework supports adversarial negative sampling, early stopping, and comprehensive evaluation metrics (MR, MRR, Hits@k).

2. Features
Multiple Embedding Models:
TransE, RotatE, HAKE, DistMult, TransD, ComplEx, CompoundE
Joint Training:
Instance-level (G-ins), type-level (G-type), and entity-type pairs (P-type)
Efficient Negative Sampling:
Self-adversarial sampling with subsampling weights
Evaluation Metrics:
Mean Rank (MR), Filtered MR, MRR, Filtered MRR, Hits@k
Early Stopping:
Monitors filtered MR or MRR with configurable patience
Model Saving & Loading:
Save/load checkpoints and embeddings

3. Project Structure
JT-KGE/
├─ README.md
├─ run.py
├─ train.py
├─ model/               ← 预训练权重/示例权重
│  └─ CompoundE/
│     └─ savefold_name/
│        ├─ ins_model.vec.pt
│        ├─ type_model.vec.pt
│        ├─ pair_model.vec.pt
│        ├─ config.npy
│        ├─ ent_labels.tsv
│        ├─ rel_labels.tsv
│        ├─ type_ent_labels.tsv
│        └─ ...
├─ sample_data/         ← 样例数据（可直接跑通）
│  ├─ train.txt
│  ├─ valid.txt
│  ├─ test.txt
│  ├─ train_type.txt
│  ├─ valid_type.txt
│  └─ test_type.txt
├─ data_helper.py
├─ model.py
├─ loss.py
├─ optimizer.py
├─ evaluator.py
├─ earlystopper.py
└─ utils.py
**
4. Installation
Requirements
Python ≥3.7
PyTorch ≥1.8
tqdm, numpy, pandas

pip install torch tqdm numpy pandas

5. Quick Start

5.1 Prepare Data
Organize your data into the following files:
**
sample_data/
├── train.txt          # Instance-level triplets (h, r, t, h_type, t_type)
├── valid.txt
├── test.txt
├── train_type.txt     # Type-level triplets (h, r, t)
├── valid_type.txt
├── test_type.txt
**
5.2 Run Training
python run.py --method CompoundE --device cuda

5.3 Key Arguments
| Argument         | Description                          | Default    |
| ---------------- | ------------------------------------ | ---------- |
| `--method`       | Embedding model (transe/rotate/hake) | `DistMult` |
| `--epochs`       | Training epochs                      | `200`      |
| `--dim_ins`      | Instance embedding dimension         | `200`      |
| `--dim_type`     | Type embedding dimension             | `200`      |
| `--batch_size_*` | Batch sizes for ins/type/pair        | `256`      |
| `--lr`           | Learning rate                        | `0.001`    |
| `--neg_rate`     | Negative samples per positive        | `5`        |
| `--patience`     | Early stopping patience              | `3`        |


6. Usage Examples
6.1 Train with RotatE
python run.py --method rotate --dim_ins 400 --lr 0.0001 --device cuda

6.2 Resume Training
python run.py --method CompoundE --load_model ./model/CompoundE/savefold_name

7. Link Prediction

| Model        | MR↓   | MRR↑  | Hits@1↑ | Hits@3↑ | Hits@5↑ | Hits@10↑ |
|--------------|-------|-------|---------|---------|---------|----------|
| **CompoundE (ours)** | **9.859** | **0.308** | **0.173** | **0.328** | **0.435** | **0.639** |
| HAKE (ours)  | 12.242 | 0.2609 | 0.137 | 0.264 | 0.349 | 0.542 |
| TransE (ours)| 14.418 | 0.262 | 0.149 | 0.262 | 0.338 | 0.521 |
| RotatE (ours)| 14.757 | 0.232 | 0.125 | 0.233 | 0.306 | 0.439 |
| DistMult (ours)| 17.422 | 0.248 | 0.151 | 0.250 | 0.329 | 0.329 |
| ComplEx (ours)| 0.164 | 0.271 | 0.176 | 0.262 | 0.349 | 0.481 |

| Model   | MR↓   | MRR↑  | Hits@1↑ | Hits@3↑ | Hits@5↑ | Hits@10↑ |
|---------|-------|-------|---------|---------|---------|----------|
| CompoundE (paper) | 15.767 | 0.255 | 0.134 | 0.255 | 0.340 | 0.537 |
| HAKE (paper)      | 13.537 | 0.262 | 0.143 | 0.277 | 0.359 | 0.503 |
| TransE (paper)    | 16.275 | 0.265 | 0.164 | 0.259 | 0.339 | 0.485 |
| RotatE (paper)    | 16.151 | 0.203 | 0.104 | 0.192 | 0.262 | 0.402 |
| DistMult (paper)  | 18.095 | 0.240 | 0.146 | 0.243 | 0.301 | 0.410 |
| ComplEx (paper)   | 15.275 | 0.263 | 0.164 | 0.265 | 0.346 | 0.468 |













