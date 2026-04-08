# Step 6: Running the MVFoul Project

This guide walks you through actually running the project from start to finish, assuming you've completed Steps 1–5.

---

## Overview: What are we doing?

We're training three different neural networks to detect **dives (simulation)** in soccer using the SoccerNet-MVFoul dataset. Each approach uses a different strategy:

| Approach | Strategy | Input | Training time |
|---|---|---|---|
| **A** | Video analysis | Raw video frames | ~6 hours |
| **B** | Pose analysis | Skeleton keypoints (body joints) | ~2 hours |
| **C** | Graph analysis (novel) | Skeleton + velocity | ~3 hours |

At the end, we compare which approach best detects dives.

---

## Step 1: Clone the repository

If you haven't already:

```bash
cd ~
mkdir -p projects
cd projects
git clone https://github.com/YOUR-ORG/var.git
cd var
```

> Ask your supervisor for the actual repository URL.

---

## Step 2: Open the project in VS Code

```bash
cd ~/projects/var
code .
```

VS Code opens with all the project files in the left panel. Open the integrated terminal with `` Ctrl+` ``.

---

## Step 3: Run the setup script

This creates your Python environment and installs all dependencies:

```bash
cd local_training
bash setup.sh
```

**What this does:**
1. Checks that Python 3 is installed
2. Checks that your NVIDIA GPU is visible
3. Creates a `.venv` virtual environment
4. Installs PyTorch with CUDA support
5. Installs all other required packages (MediaPipe, scikit-learn, etc.)
6. Runs a quick verification to confirm everything works

**Expected output at the end:**

```
=== Verification ===
PyTorch     : 2.x.x
CUDA avail  : True
  GPU 0: NVIDIA RTX A4000  VRAM: 16.0 GB
  GPU 0: matmul OK
MediaPipe   : 0.x.x
OpenCV      : 4.x.x
scikit-learn: 1.x.x
=== Setup complete ===
```

If CUDA shows `False`, go back to [Step 5: CUDA Setup](05-cuda-gpu-setup.md).

---

## Step 4: Activate the virtual environment

**You must do this every time you open a new terminal:**

```bash
source .venv/bin/activate
```

You'll see `(.venv)` at the start of your terminal prompt. If you don't see it, the environment isn't active and Python won't find the installed packages.

---

## Step 5: Download the dataset

Edit `get_data.sh` first to set your SoccerNet password:

```bash
nano get_data.sh
```

Find the line with `SOCCERNET_PASSWORD` and set it to your password (your supervisor will provide this). Press `Ctrl+O` to save, `Ctrl+X` to exit nano.

Then run:

```bash
bash get_data.sh
```

This downloads ~50 GB of video data. It will take a while depending on your internet connection.

After downloading, update `config.yaml` to point to your data:

```bash
nano src/config.yaml
```

Change the `data.root` path to wherever `get_data.sh` put the dataset.

---

## Step 6: Run a smoke test (always do this first!)

Before committing to an 11-hour training run, verify everything works on a tiny subset:

```bash
bash run_all.sh --debug
```

This trains all three approaches on just 60 actions (~25 minutes total). If it completes without errors, you're ready for the full run.

### What to look for:

- Each approach should start, show a progress bar, and finish
- You should see metrics printed at the end (balanced accuracy, Dive F1)
- No errors or crashes

---

## Step 7: Full training run

Once the smoke test passes:

```bash
bash run_all.sh
```

This trains:
1. **Approach B** first (~2 hours) — also builds the pose cache
2. **Approach C** second (~3 hours) — reuses the pose cache
3. **Approach A** last (~6 hours) — heaviest, uses raw video

**Total: ~11 hours.** Start it before bed and let it run overnight.

> **Tip:** You can safely close VS Code while training runs in the terminal. The process continues as long as the WSL terminal stays open. If you want to make sure it survives even if the terminal closes, use:
> ```bash
> nohup bash run_all.sh > training_output.log 2>&1 &
> ```

---

## Step 8: Monitor training (optional but recommended)

### Option A: Watch the terminal

The training script prints progress to the terminal — you can just watch it.

### Option B: TensorBoard (graphical)

Open a **second terminal** (click the `+` in VS Code's terminal panel):

```bash
cd ~/projects/var/local_training
source .venv/bin/activate
tensorboard --logdir results/tensorboard --port 6006
```

Then open your web browser (on Windows) and go to:

```
http://localhost:6006
```

TensorBoard shows live graphs of:
- Training loss (should go down)
- Balanced accuracy (should go up)
- Dive F1 score (should go up)

### Option C: Monitor GPU usage

In another terminal:

```bash
watch -n 2 nvidia-smi
```

---

## Step 9: Check results

After training completes, results are in `local_training/results/`:

```
results/
├── approach_A_metrics.json      # Final metrics for Approach A
├── approach_B_metrics.json      # Final metrics for Approach B
├── approach_C_metrics.json      # Final metrics for Approach C
├── approach_C_joint_importance.csv  # Which body joints matter for dive detection
└── tensorboard/                 # Training curves
```

View the metrics:

```bash
cat results/approach_A_metrics.json
cat results/approach_B_metrics.json
cat results/approach_C_metrics.json
```

### What the metrics mean

| Metric | What it measures | Good values |
|---|---|---|
| **Balanced Accuracy** | Average accuracy across all classes, weighted equally | > 0.5 is decent, > 0.7 is good |
| **Dive F1** | How well the model specifically detects dives | > 0.3 is a start, > 0.5 is good |
| **Macro F1** | Average F1 across all classes | > 0.4 is decent |

> **Never use raw accuracy** — because dives are only ~0.9% of the data, a model that always predicts "not dive" gets >99% accuracy while being completely useless.

---

## Running individual approaches

If you want to run just one approach:

```bash
source .venv/bin/activate

# Just Approach B
python3 src/train.py --approach B --config src/config.yaml

# Just Approach C with debug mode
python3 src/train.py --approach C --config src/config.yaml --debug

# Just Approach A with smaller batch size (if you run out of GPU memory)
python3 src/train.py --approach A --config src/config.yaml --batch_size 4
```

---

## Editing configuration

All hyperparameters are in `src/config.yaml`. Open it in VS Code to edit.

Key settings you might want to change:

| Setting | What it controls | Default |
|---|---|---|
| `data.root` | Where the dataset is stored | `/data/mvfoul` |
| `training.debug_mode` | Use tiny dataset for testing | `false` |
| `approach_A.batch_size` | Batch size for video model | `8` |
| `approach_B.n_epochs` | How many times to loop through data | `50` |
| `approach_C.lr` | Learning rate (how fast the model learns) | `0.001` |

> **Rule of thumb:** Only change one thing at a time, and always run a debug test first.

---

## Common issues

### Training freezes or hangs

- Check GPU memory: `nvidia-smi` — if VRAM is 100% full, reduce batch size
- Check `num_workers` in config.yaml — keep it at 2 or lower in WSL2
- Make sure data is on the Linux filesystem, not `/mnt/c/`

### "Killed" message (no other error)

Your system ran out of RAM (not GPU memory — regular RAM). Close other applications, especially web browsers with many tabs.

### Training loss is NaN

The learning rate may be too high. Try reducing it in `config.yaml`.

### Low Dive F1 (close to 0)

This is expected in early epochs. The model needs time to learn the rare dive class. If it's still 0 after many epochs, the class weights may need adjustment.

---

## Quick reference

```bash
# Every time you start working:
cd ~/projects/var/local_training
source .venv/bin/activate

# Smoke test:
bash run_all.sh --debug

# Full training:
bash run_all.sh

# Monitor:
tensorboard --logdir results/tensorboard --port 6006

# Check results:
cat results/approach_B_metrics.json
```

---

## Congratulations!

If you've followed all six steps, you've:
1. Set up WSL2 (a Linux environment on Windows)
2. Configured VS Code for development
3. Learned Python basics and virtual environments
4. Learned Git for version control
5. Set up NVIDIA GPU/CUDA support
6. Run the full MVFoul training pipeline

Questions? Reach out to your supervisor.
