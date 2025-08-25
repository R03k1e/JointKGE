## Repository Overview
This project implements the JT-KGE framework for jointly training geographic knowledge graph embedding models. It supports multiple embedding methods (TransE, RotatE, HAKE, CompoundE, DistMult, TransD, ComplEx) and provides end-to-end data processing, training, and evaluation pipelines for geographic KG alignment tasks.

The repository contains two main components:
1. **A multi-model joint-training framework for geographic KG embeddings**
2. **A plug-and-play tool for entity-relation inference and visualization**

## Environment Setup
All runtime dependencies are listed in `requirements.txt`. Install them with:
```bash
pip install -r requirements.txt
```

## Dataset
- The dataset is located in the `sample data` folder and contains triples for geographic KG embedding.

## Running the Code

### Multi-Model Joint-Training Framework
- **Entry point**: `run.py`  
- One-command start for training and testing; model type, dimension, batch size, etc. can be set via CLI.  
- After training, model weights, embeddings, logs, and evaluation results are automatically saved under `./model/`.

### Plug-and-Play Entity-Relation Inference & Visualization
- **Inference script**: `testRel.py`  
- Copy the trained `ent.tsv`, `rel.tsv`, `ent_labels.tsv`, and `rel_labels.tsv` into the same directory, then run the script to interactively input any entity pair and obtain the Top-K predicted relations in real time.  
- No re-training required—useful for demos, quick validation, or downstream integration.

## Experimental Procedures

### ① Multi-Model Performance Comparison (Table 4 Link-Prediction Results)
1. Open `run.py` and set `--method` to CompoundE, HAKE, TransE, RotatE, DistMult, and ComplEx in turn, all using the train/valid/test triples in `sample data`.
2. Run  
   ```bash
   python run.py --method CompoundE --epochs 200 --lr 0.001 --dim_ins 200 --neg_rate 5
   ```  
   Wait for training to finish; results are automatically generated in `./model/CompoundE/savefold_name/`.
3. Repeat step 2 for all models, collect **MR, MRR, Hits@1/3/5/10** on the test set, and fill them into Table 4.

### ② Entity-Relation Direction Visualization (Tables 5 & 6)
1. Copy the trained `ent.tsv`, `rel.tsv`, `ent_labels.tsv`, `rel_labels.tsv` into the same directory as `testRel.py`.
2. Run  
   ```bash
   python testRel.py
   ```
3. When prompted, enter the entity pairs:  
   - “Jiaozhou City & Jiali County”  
   - “Yongsheng County & Suijiang County”  
   - “Quanshan District & Lanling County”  
4. The script outputs the Top-3 predicted directions (e.g., South-Southwest-West). Compare the results with baseline models (HAKE-base, CompoundE-base).

### ③ Varying Training-Set Ratio Comparison (Table 7)
1. Four pre-split training sets are provided in the same directory as `run.py`:
   - `./sample data/train_10.txt`
   - `./sample data/train_30.txt`
   - `./sample data/train_50.txt`
   - `./sample data/train_80.txt`

2. For the 10 % example, overwrite the original file:  
   ```bash
   cp sample\ data/train_10.txt sample\ data/train.txt
   ```

3. Run (any model; here we use CompoundE):  
   ```bash
   python run.py --method CompoundE --epochs 200 --lr 0.001 --dim_ins 200 --neg_rate 5
   ```  
   After training, record **MR, MRR, Hits@10** on the test set.

4. Repeat steps 2–3 for `train_30/50/80.txt` and summarize the four sets of metrics.
