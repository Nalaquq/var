# MVFoul — Local Training Setup

Three-approach comparison for dive/simulation detection on SoccerNet-MVFoul.  
All three approaches run sequentially on a single **RTX A4000 (16 GB)** laptop.

## Expected training times (full dataset, RTX A4000)

| Order | Approach | Est. time | Notes |
|---|---|---|---|
| 1st | **B** — BiLSTM pose | ~2 hrs | Runs first; builds pose cache reused by C |
| 2nd | **C** — ST-GCN novel | ~3 hrs | Reuses pose cache, adds velocity features |
| 3rd | **A** — MViT video | ~6 hrs | Heaviest; run overnight |
| | **Total** | ~11 hrs | Run `bash run_all.sh` and step away |

Debug mode (60-action subset) takes ~25 min total — validate the pipeline first.

---

## First-time setup

```bash
# 1. Extract the project
tar -xzf mvfoul_project.tar.gz
cd mvfoul_project

# 2. Set up the virtual environment (detects CUDA version automatically)
bash setup.sh

# 3. Activate
source .venv/bin/activate

# 4. Download the dataset (edit SOCCERNET_PASSWORD and DATA_DIR in get_data.sh first)
bash get_data.sh

# 5. Edit config.yaml — set data.root to wherever you put the dataset
nano src/config.yaml
```

---

## Running

**Recommended: run everything with one command and step away**
```bash
source .venv/bin/activate
bash run_all.sh
```

**Or run each approach individually:**
```bash
python src/train.py --approach B --config src/config.yaml   # ~2 hrs
python src/train.py --approach C --config src/config.yaml   # ~3 hrs
python src/train.py --approach A --config src/config.yaml   # ~6 hrs
```

**Quick smoke test first (always do this before the full run):**
```bash
python src/train.py --approach C --config src/config.yaml --debug
# Should complete in ~8 min and confirm the full pipeline works end-to-end
```

**Out of VRAM on Approach A? Override batch size without editing the config:**
```bash
python src/train.py --approach A --config src/config.yaml --batch_size 4
```

---

## Monitoring

```bash
tensorboard --logdir results/tensorboard --port 6006
# Open http://localhost:6006
```

TensorBoard tracks per epoch: `train/loss`, `train/balanced_acc`,
`valid/balanced_acc`, `valid/dive_f1`.

---

## Output files

```
results/
  approach_A_history.csv          # per-epoch metrics
  approach_A_metrics.json         # final validation metrics
  approach_B_history.csv
  approach_B_metrics.json
  approach_C_history.csv
  approach_C_metrics.json
  approach_C_joint_importance.csv # which joints drive Dive detection
  tensorboard/approach_A/
  tensorboard/approach_B/
  tensorboard/approach_C/
  run_all.log                     # full training log

checkpoints/
  approach_A_best.pt              # best checkpoint (by balanced accuracy)
  approach_B_best.pt
  approach_C_best.pt

feature_cache/   (set in config.yaml as data.cache_dir)
  pose_<action_id>.npy            # MediaPipe keypoints, extracted once
  graph_<action_id>.npy           # velocity-augmented graph features
```

The pose cache is shared between Approaches B and C — MediaPipe extraction
only runs once (~20 min for the full dataset), then both approaches read from disk.

---

## Key metrics

**Primary — Balanced Accuracy**: average per-class recall. Matches the VARS paper.  
**Secondary — Dive F1**: whether the model actually learns the dive class.  
**Never use raw accuracy**: a model that always predicts "not dive" scores >99%.

---

## VRAM guide (RTX A4000, 16 GB)

| Approach | Batch size | VRAM usage | Notes |
|---|---|---|---|
| A (MViT, head-only) | 8 | ~6 GB | Safe |
| A (MViT, fine-tune) | 8 | ~13 GB | If OOM, use `--batch_size 4` |
| B (BiLSTM) | 128 | ~2 GB | Tiny model |
| C (ST-GCN) | 128 | ~3 GB | Tiny model |

---

## Project structure

```
mvfoul_project/
├── setup.sh           # one-time environment setup
├── get_data.sh        # dataset download (direct from SoccerNet or via Drive)
├── run_all.sh         # train all three approaches sequentially
├── README.md
└── src/
    ├── config.yaml    # all hyperparameters — edit here, not in code
    ├── train.py       # CLI entry point (--approach A/B/C)
    ├── models.py      # ApproachA_MViT, ApproachB_BiLSTM, ApproachC_STGCN
    └── dataset.py     # data loading, pose extraction, caching, graph building
```
