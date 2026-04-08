# MVFoul — Complete Project Package

Dive/simulation detection in soccer using SoccerNet-MVFoul.
Student: Manning Lasso | Supervisor: Sean (Nalaquq LLC)

---

## What's in here

```
full_project/
│
├── README.md                          ← you are here
│
├── literature/
│   └── VAR_Literature_Review_and_Project_Scope.docx
│       Full literature review (5 clusters, 10 key papers) +
│       scoped project plan with 8-week timeline and risk table.
│
├── notebooks/                         ← run these in Google Colab
│   ├── 01_dataset_statistics.ipynb
│   │   Phase 0: download SoccerNet-MVFoul annotations, parse JSON,
│   │   generate summary statistics and 7 plots focused on Dive class
│   │   distribution, contact rates, severity, and class weights.
│   │   Run this first to understand the data before training anything.
│   │
│   └── 02_three_baselines.ipynb
│       Phase 1–3 in one notebook: all three model architectures
│       side-by-side on identical data. Runs in Colab (T4 GPU).
│       DEBUG_MODE=True by default (~10 min); set False for full run.
│
└── local_training/                    ← run these locally in WSL + VS Code
    ├── README.md                      local-specific setup instructions
    ├── setup.sh                       creates .venv, installs all deps
    ├── get_data.sh                    downloads dataset from SoccerNet
    ├── run_all.sh                     trains B → C → A sequentially
    └── src/
        ├── config.yaml                all hyperparameters (edit here)
        ├── train.py                   CLI: --approach A/B/C --debug
        ├── models.py                  ApproachA_MViT, B_BiLSTM, C_STGCN
        └── dataset.py                 data loading, pose extraction, caching
```

---

## Recommended workflow

### Step 1 — Understand the data (Colab, ~30 min)
Open `notebooks/01_dataset_statistics.ipynb` in Google Colab.
Run all cells. Key outputs:
- Dive class is ~0.9% of actions (severe imbalance — weighted loss is essential)
- Contact rate for dives vs. genuine fouls (tests the core hypothesis)
- Class weights to paste into training config

### Step 2 — Validate the pipeline (Colab or local, ~25 min)
Open `notebooks/02_three_baselines.ipynb`. With `DEBUG_MODE=True`
(default), all three architectures run end-to-end on 60 actions.
Confirms data loading, model forward passes, and metric computation
before committing to a full training run.

### Step 3 — Full training run (local WSL, ~11 hrs on RTX A4000)
```bash
cd local_training
bash setup.sh                   # one-time: creates .venv, installs deps
source .venv/bin/activate
bash get_data.sh                # one-time: downloads dataset (~50 GB)
bash run_all.sh --debug         # smoke test first (~25 min)
bash run_all.sh                 # full run, leave overnight
```

### Step 4 — Monitor
```bash
tensorboard --logdir local_training/results/tensorboard --port 6006
```

### Step 5 — Compare results
Results land in `local_training/results/`:
- `approach_*_metrics.json`     — balanced accuracy, Dive F1, macro F1
- `approach_C_joint_importance.csv` — which skeleton joints drive Dive detection
- `tensorboard/`               — learning curves for all three approaches

---

## The three approaches at a glance

| | Approach | Architecture | Reference |
|---|---|---|---|
| A | Video baseline | MViTv2-S pretrained on Kinetics-400 | VARS (Held et al. 2023) |
| B | Pose + BiLSTM | MediaPipe 33-kp → Bidirectional LSTM | Fang et al. 2024 |
| C | **ST-GCN (novel)** | Velocity-augmented skeleton graph + ST-GCN | This project |

The novel contribution (C) adds velocity [vx, vy] as node features
alongside position [x, y], encoding the contact-response signature
that distinguishes genuine falls from pre-planned dives.

---

## Key metric: Balanced Accuracy and Dive F1
Do not use raw accuracy — a model that always predicts "not dive"
scores >99% due to class imbalance. Balanced Accuracy and Dive F1
are the only meaningful metrics for this problem.
