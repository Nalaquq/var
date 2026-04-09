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
│   ├── 00_unified_pipeline.ipynb      ★ START HERE
│   │   All-in-one notebook: dataset statistics, plots, and all three
│   │   model architectures in a single file. No file transfers needed.
│   │   DEBUG_MODE=True by default (~15 min); set False for full run.
│   │
│   ├── 01_dataset_statistics.ipynb    (original, kept for reference)
│   ├── 02_three_baselines.ipynb       (original, kept for reference)
│   │
│   └── outputs/                       ← CSV outputs from Colab notebooks
│       Downloaded after each notebook run. See docs/07-google-colab.md.
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

## Getting started (new to this?)

If you're new to Python, WSL, VS Code, Git, or CUDA, start here. These step-by-step tutorials walk you through everything from scratch:

1. [Installing WSL2](docs/01-wsl-setup.md) — Get a Linux environment running on Windows
2. [Setting Up VS Code](docs/02-vscode-setup.md) — Install and connect VS Code to WSL
3. [Python and Virtual Environments](docs/03-python-basics.md) — Python basics, pip, and venvs
4. [Git and GitHub](docs/04-git-github.md) — Version control fundamentals
5. [NVIDIA Drivers and CUDA](docs/05-cuda-gpu-setup.md) — GPU setup for deep learning
6. [Running the MVFoul Project](docs/06-running-the-project.md) — End-to-end walkthrough of training
7. [Using Google Colab](docs/07-google-colab.md) — Uploading/downloading files, running notebooks in Colab

Complete them in order — each one builds on the previous (Step 7 can be done anytime).

---

## Recommended workflow

### Step 1 — Run the unified notebook (Colab, ~15 min)
Open `notebooks/00_unified_pipeline.ipynb` in Google Colab (see [docs/07-google-colab.md](docs/07-google-colab.md) for step-by-step instructions). Make sure to select a **T4 GPU** runtime.

Run all cells. With `DEBUG_MODE=True` (default), everything runs end-to-end in ~15 minutes. You will get:
- Dataset statistics and 7 plots (class imbalance, contact rates, severity, etc.)
- All three model architectures trained and compared side-by-side
- CSV exports and training results to download

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
