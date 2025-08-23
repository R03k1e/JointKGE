# JT-KGE: A Joint Knowledge Graph Embedding Framework

**JT-KGE** (Joint Type-based Knowledge Graph Embedding) is a flexible and extensible PyTorch framework for learning representations of entities, relations, and types in Knowledge Graphs (KGs). It supports multiple embedding models (e.g., TransE, RotatE, HAKE, DistMult, TransD, ComplEx, CompoundE) and jointly trains instance-level and type-level information.

---

## 1. Overview

JT-KGE is designed for **jointly learning**:
- **Instance-level embeddings**: Capturing relations between entities.
- **Type-level embeddings**: Capturing hierarchical types of entities.
- **Entity-type associations**: Linking entities to their semantic types.

The framework supports **adversarial negative sampling**, **early stopping**, and **comprehensive evaluation metrics** (MR, MRR, Hits@k).

---

## 2. Features

- **Multiple Embedding Models**:
  - TransE, RotatE, HAKE, DistMult, TransD, ComplEx, CompoundE
- **Joint Training**:
  - Instance-level (`G-ins`), type-level (`G-type`), and entity-type pairs (`P-type`)
- **Efficient Negative Sampling**:
  - Self-adversarial sampling with subsampling weights
- **Evaluation Metrics**:
  - Mean Rank (MR), Filtered MR, MRR, Filtered MRR, Hits@k
- **Early Stopping**:
  - Monitors filtered MR or MRR with configurable patience
- **Model Saving & Loading**:
  - Save/load checkpoints and embeddings

---

## 3. Project Structure

```
JT-KGE/
├── data_helper.py      # Data loading and preprocessing
├── model.py            # Embedding models (TransE, RotatE, etc.)
├── loss.py             # Loss functions (MarginLoss, LogisticLoss)
├── optimizer.py        # Optimizer wrapper (Adam, Adagrad, SGD)
├── evaluator.py        # Evaluation metrics and testing
├── earlystopper.py     # Early stopping logic
├── train.py            # Training loop and trainer classes
├── run.py              # Entry point for experiments
├── utils.py            # Utilities (e.g., Monitor enum)
└── README.md           # This file
```

---

## 4. Installation

### Requirements
- Python ≥3.7
- PyTorch ≥1.8
- tqdm, numpy, pandas

### Install
```bash
pip install torch tqdm numpy pandas
```

---

## 5. Quick Start

### 5.1 Prepare Data
Organize your data into the following files:
```
sample_data/
├── train.txt          # Instance-level triplets (h, r, t, h_type, t_type)
├── valid.txt
├── test.txt
├── train_type.txt     # Type-level triplets (h, r, t)
├── valid_type.txt
├── test_type.txt
```

### 5.2 Run Training
```bash
python run.py --method CompoundE --device cuda
```

### 5.3 Key Arguments
| Argument         | Description                          | Default |
|------------------|--------------------------------------|---------|
| `--method`       | Embedding model (transe/rotate/hake) | `DistMult` |
| `--epochs`       | Training epochs                      | `200`   |
| `--dim_ins`      | Instance embedding dimension         | `200`   |
| `--dim_type`     | Type embedding dimension             | `200`   |
| `--batch_size_*` | Batch sizes for ins/type/pair        | `256`   |
| `--lr`           | Learning rate                        | `0.001` |
| `--neg_rate`     | Negative samples per positive        | `5`     |
| `--patience`     | Early stopping patience              | `3`     |

---

## 6. Usage Examples

### 6.1 Train with RotatE
```bash
python run.py --method rotate --dim_ins 400 --lr 0.0001 --device cuda
```

### 6.2 Resume Training
```bash
python run.py --method CompoundE --load_model ./model/CompoundE/savefold_name
```

### 6.3 Evaluate Saved Model
```python
from train import Join_trainer
from data_helper import GeoKG

args = GeoKGArgparse(['--method', 'HAKE', '--device', 'cuda'])
trainer = Join_trainer(args)
trainer.build(GeoKG())
trainer.evaluator.full_test()
```

---

## 7. Output Files

After training, the following are saved in `./model/{method}/{save_times}/`:
- `ins_model.vec.pt`: Instance-level model
- `type_model.vec.pt`: Type-level model
- `pair_model.vec.pt`: Entity-type model
- `ent_labels.tsv`: Entity ID mappings
- `rel_labels.tsv`: Relation ID mappings
- `type_*_labels.tsv`: Type ID mappings
- `*_Training_results_*.csv`: Training losses
- `*_Eval_results_*.csv`: Evaluation metrics
- `config.npy`: Training arguments

---

## 8. Extending the Framework

### 8.1 Add a New Model
1. Implement a new class in `model.py` (e.g., `ins_model_new`, `type_model_new`).
2. Add initialization in `Join_trainer.build()`.

### 8.2 Custom Loss Function
1. Add a new loss class in `loss.py` inheriting from `Loss`.
2. Override `forward()` method.

### 8.3 New Evaluation Metrics
1. Modify `evaluator.py` to compute additional metrics.
2. Update `Metric.settle()` and `display_summary()`.

---

## 9. Citation
If you use this framework, please cite:
```
@misc{jt-kge,
  title={JT-KGE: A Joint Type-based Knowledge Graph Embedding Framework},
  author={Your Name},
  year={2023},
  url={https://github.com/your-repo/JT-KGE}
}
```

---

## 10. License
MIT License - see [LICENSE](LICENSE) for details.

---

## 11. Support
For issues or questions, open a GitHub issue or contact `your-email@example.com`.
