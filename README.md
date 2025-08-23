# JT-KGE: A Joint Knowledge-Graph Embedding Framework

**JT-KGE** (Joint Type-based Knowledge Graph Embedding) is a flexible PyTorch framework for learning entity, relation and type representations in Knowledge Graphs. It supports **TransE, RotatE, HAKE, DistMult, TransD, ComplEx** and **CompoundE**, and performs **joint training** over instance-level triples, type-level triples and entity-type pairs.

---

## 1. Overview
- **Instance-level embeddings** (G-ins): relations between entities  
- **Type-level embeddings** (G-type): taxonomic relations between types  
- **Entity–type pairs** (P-type): membership of entities in semantic types  

Includes adversarial negative sampling, early stopping and comprehensive link-prediction metrics (MR, MRR, Hits@k).

---

## 2. Features
- **7 embedding models** out-of-the-box  
- **Joint training** with tunable weighting  
- **Self-adversarial negative sampling**  
- **Early stopping** on filtered MR or MRR  
- **Automatic checkpointing & embedding export**

---

## 3. Repository Structure
```
JT-KGE/
├── data_helper.py   # data loading & indexing
├── model.py         # all embedding models
├── loss.py          # MarginLoss / LogisticLoss
├── optimizer.py     # Adam / Adagrad / SGD wrapper
├── evaluator.py     # link-prediction metrics
├── earlystopper.py  # patience-based early stop
├── train.py         # training loop
├── run.py           # CLI entry point
├── utils.py
├── model/           # pre-trained weights (optional)
├── sample_data/     # toy dataset
├── results/         # experimental logs
└── README.md
```

---

## 4. Installation
```bash
pip install torch tqdm numpy pandas
```

---

## 5. Quick Start

### 5.1 Data Format  
Place files in `sample_data/`:

```
sample_data/
├── train.txt          # h r t h_type t_type
├── valid.txt
├── test.txt
├── train_type.txt     # h r t  (type-level)
├── valid_type.txt
└── test_type.txt
```

### 5.2 Train a Model
```bash
python run.py --method CompoundE --device cuda
```

### 5.3 Key Arguments
| Flag            | Description                 | Default |
|-----------------|-----------------------------|---------|
| `--method`      | transe / rotate / hake / ... | DistMult |
| `--epochs`      | training epochs             | 200     |
| `--dim_ins`     | entity/relation dim         | 200     |
| `--dim_type`    | type dim                    | 200     |
| `--lr`          | learning rate               | 0.001   |
| `--neg_rate`    | negative samples / pos      | 5       |
| `--patience`    | early-stop patience         | 3       |

---

## 6. Usage Examples
```bash
# Train RotatE
python run.py --method rotate --dim_ins 400 --lr 0.0001 --device cuda

# Resume from checkpoint
python run.py --method CompoundE --load_model ./model/CompoundE/savefold_name
```

---

## 7. Output Files
After training, `./model/{method}/{save_times}/` contains:
- `ins_model.vec.pt`, `type_model.vec.pt`, `pair_model.vec.pt`  
- `*_labels.tsv` (id → name)  
- `*_Training_results_*.csv` & `*_Eval_results_*.csv`  
- `config.npy`

---

## 8. Extending the Framework
1. **New model**: inherit `nn.Module` in `model.py` and register in `Join_trainer.build()`.  
2. **Custom loss**: subclass `Loss` in `loss.py`.  
3. **New metrics**: modify `evaluator.py`.

---

## 9. Experimental Results

Link-prediction on the provided benchmark:

| Model               | MR↓   | MRR↑  | Hits@1↑ | Hits@3↑ | Hits@5↑ | Hits@10↑ |
|---------------------|-------|-------|---------|---------|---------|----------|
| **CompoundE (ours)**| **9.86** | **0.308** | **0.173** | **0.328** | **0.435** | **0.639** |
| HAKE (ours)         | 12.24 | 0.261 | 0.137 | 0.264 | 0.349 | 0.542 |
| TransE (ours)       | 14.42 | 0.262 | 0.149 | 0.262 | 0.338 | 0.521 |
| RotatE (ours)       | 14.76 | 0.232 | 0.125 | 0.233 | 0.306 | 0.439 |
| DistMult (ours)     | 17.42 | 0.248 | 0.151 | 0.250 | 0.329 | 0.329 |
| ComplEx (ours)      | 0.16  | 0.271 | 0.176 | 0.262 | 0.349 | 0.481 |

Comparison with original baselines:

| Model        | MR↓   | MRR↑  | Hits@1↑ | Hits@3↑ | Hits@5↑ | Hits@10↑ |
|--------------|-------|-------|---------|---------|---------|----------|
| CompoundE    | 15.77 | 0.255 | 0.134 | 0.255 | 0.340 | 0.537 |
| HAKE         | 13.54 | 0.262 | 0.143 | 0.277 | 0.359 | 0.503 |
| TransE       | 16.28 | 0.265 | 0.164 | 0.259 | 0.339 | 0.485 |
| RotatE       | 16.15 | 0.203 | 0.104 | 0.192 | 0.262 | 0.402 |
| DistMult     | 18.10 | 0.240 | 0.146 | 0.243 | 0.301 | 0.410 |
| ComplEx      | 15.28 | 0.263 | 0.164 | 0.265 | 0.346 | 0.468 |

> Bold numbers denote best performance in our reproduction.  
> “(ours)” indicates results reproduced and fine-tuned within JT-KGE.

---

