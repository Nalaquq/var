# Step 5: NVIDIA Drivers and CUDA for WSL2

Training deep learning models (like the ones in this project) on a CPU would take days. A GPU (graphics card) can do the same work in hours. This guide helps you set up your NVIDIA GPU to work inside WSL2.

---

## What is CUDA?

CUDA is NVIDIA's software platform that lets programs run computations on the GPU. PyTorch (the deep learning library we use) talks to your GPU through CUDA.

The chain looks like this:

```
Your Python code
    → PyTorch
        → CUDA libraries
            → NVIDIA driver
                → Your GPU hardware (e.g., RTX A4000)
```

---

## Important: Driver goes on Windows, not Linux

This is the most common mistake. In WSL2:

- **NVIDIA driver** → Install on **Windows** (the host)
- **CUDA toolkit** → You do NOT need to install this separately. PyTorch brings its own CUDA libraries.

> **Do NOT install NVIDIA drivers inside WSL/Linux.** The Windows driver is automatically shared with WSL2.

---

## Step 1: Install (or update) the NVIDIA driver on Windows

1. Go to [https://www.nvidia.com/drivers](https://www.nvidia.com/drivers)
2. Select your GPU model (e.g., RTX A4000), Operating System: **Windows 10/11 64-bit**
3. Download and install the **Game Ready** or **Studio** driver (either works)
4. Restart your computer after installation

Alternatively, if you already have GeForce Experience or NVIDIA App installed, open it and check for driver updates.

### Minimum driver version

For CUDA 12.x support (recommended), you need driver version **525.60 or newer**. Most recent drivers are well above this.

---

## Step 2: Verify the GPU is visible in WSL

Open your Ubuntu terminal and run:

```bash
nvidia-smi
```

You should see something like:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A4000    Off  | 00000000:01:00.0 Off |                  Off |
| 41%   34C    P8    14W / 140W |    123MiB / 16376MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

Key things to check:
- **Driver Version** — should be 525+ (the number at the top)
- **CUDA Version** — should be 11.x or 12.x
- **GPU Name** — should show your actual GPU
- **Memory** — shows total VRAM (e.g., 16376 MiB ≈ 16 GB)

---

## Step 3: Verify PyTorch can see the GPU

After setting up your virtual environment (see [Step 3](03-python-basics.md)), run:

```bash
cd ~/projects/var/local_training
source .venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```
True
NVIDIA RTX A4000
```

If it says `False`, see Troubleshooting below.

---

## Understanding GPU memory (VRAM)

Your GPU has its own memory (VRAM), separate from your computer's RAM. Models and data must fit in VRAM during training.

| What uses VRAM | Approximate size |
|---|---|
| The model weights | 100 MB – 2 GB |
| A batch of training data | Depends on batch size |
| Gradients and optimizer state | ~2x model size |
| PyTorch overhead | ~500 MB |

For this project on an RTX A4000 (16 GB VRAM):

| Approach | Batch size | VRAM usage |
|---|---|---|
| A (MViT video) | 8 | ~6–13 GB |
| B (BiLSTM) | 128 | ~2 GB |
| C (ST-GCN) | 128 | ~3 GB |

### If you run out of VRAM (OOM error)

The error looks like: `RuntimeError: CUDA out of memory`

Fix: reduce the batch size:

```bash
python3 src/train.py --approach A --config src/config.yaml --batch_size 4
```

---

## Monitoring GPU usage during training

While training is running, open a second terminal and run:

```bash
watch -n 2 nvidia-smi
```

This refreshes every 2 seconds and shows:
- **GPU-Util** — should be high (80–100%) during training
- **Memory-Usage** — how much VRAM is being used
- **Temperature** — should stay below 85°C

Press `Ctrl+C` to stop watching.

---

## Troubleshooting

### `nvidia-smi` not found in WSL

- Make sure you installed the NVIDIA driver on **Windows** (not inside Linux)
- Make sure you're running **WSL2** (not WSL1): `wsl --list --verbose`
- Restart WSL: open PowerShell and run `wsl --shutdown`, then reopen Ubuntu

### `torch.cuda.is_available()` returns `False`

1. Check `nvidia-smi` works first
2. Make sure PyTorch was installed with CUDA support. Run:
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ```
   It should print a CUDA version (like `12.1`), not `None`.
3. If it says `None`, reinstall PyTorch with CUDA. The `setup.sh` script does this automatically:
   ```bash
   rm -rf .venv
   bash setup.sh
   ```

### GPU is visible but training is slow

- Check GPU utilization with `nvidia-smi`. If GPU-Util is low (< 50%), the bottleneck is data loading, not the GPU.
- Try increasing `num_workers` in `config.yaml` (but keep it ≤ 4 in WSL2).
- Make sure your data is on the Linux filesystem, not `/mnt/c/`.

### "CUDA error: device-side assert triggered"

This usually means a bug in the code (e.g., a label index out of range), not a GPU setup issue. Share the full error with your supervisor.

---

## Key takeaways

1. Install the NVIDIA driver on **Windows**, not inside WSL
2. `nvidia-smi` confirms the GPU is visible
3. `torch.cuda.is_available()` confirms PyTorch can use it
4. If you run out of VRAM, reduce the batch size
5. Never install CUDA toolkit manually — PyTorch bundles its own

---

## Next step

Everything is set up! Move on to [Step 6: Running the MVFoul Project](06-running-the-project.md) to start training models.
