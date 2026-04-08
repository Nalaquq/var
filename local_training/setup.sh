#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — WSL2 + pip virtual environment setup for MVFoul
# Run once from the project root: bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=== MVFoul environment setup (WSL2 + pip) ==="
echo "Project : $PROJECT_DIR"
echo "Venv    : $VENV_DIR"
echo ""

# ── 0. System prerequisites ───────────────────────────────────────────────────
echo "Checking system prerequisites..."

if ! command -v python3 &>/dev/null; then
    echo "Installing python3..."
    sudo apt-get update -qq && sudo apt-get install -y python3 python3-pip python3-venv
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo "  Make sure your NVIDIA driver is installed on the Windows side"
    echo "  and you are running WSL2 (not WSL1)."
    echo "  Check with: wsl.exe --list --verbose"
    exit 1
fi

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Detect CUDA version to pick the right PyTorch wheel
CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
echo "Driver reports CUDA: $CUDA_VER"
if [[ "$CUDA_VER" == 11.* ]]; then
    TORCH_CUDA="cu118"
elif [[ "$CUDA_VER" == 12.* ]]; then
    TORCH_CUDA="cu121"
else
    echo "WARNING: unrecognised CUDA version $CUDA_VER, defaulting to cu121"
    TORCH_CUDA="cu121"
fi
echo "PyTorch CUDA build : $TORCH_CUDA"
echo ""

# ── 1. Create virtual environment ─────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at .venv"
    echo "To start fresh: rm -rf .venv && bash setup.sh"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# ── 2. Install PyTorch with CUDA ──────────────────────────────────────────────
echo "Installing PyTorch ($TORCH_CUDA)..."
pip install torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    --quiet

# ── 3. Install project dependencies ──────────────────────────────────────────
echo "Installing project dependencies..."
pip install \
    pytorchvideo \
    mediapipe \
    ultralytics \
    SoccerNet \
    av \
    opencv-python \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    tensorboard \
    pyyaml \
    tqdm \
    --quiet

# ── 4. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python - <<'PYEOF'
import torch
print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  VRAM: {p.total_memory / 1e9:.1f} GB")
        x = torch.randn(64, 64, device=f'cuda:{i}')
        _ = x @ x.T
        print(f"  GPU {i}: matmul OK")
else:
    print("WARNING: CUDA not available — check driver/WSL2 setup")

import mediapipe; print(f"MediaPipe   : {mediapipe.__version__}")
import cv2;       print(f"OpenCV      : {cv2.__version__}")
import sklearn;   print(f"scikit-learn: {sklearn.__version__}")
PYEOF

# ── 5. VS Code integration hint ───────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment in your terminal:"
echo "  source .venv/bin/activate"
echo ""
echo "VS Code — select the interpreter:"
echo "  Ctrl+Shift+P → 'Python: Select Interpreter'"
echo "  Choose: .venv/bin/python  (it should appear automatically)"
echo ""
echo "To run:"
echo "  source .venv/bin/activate"
echo "  bash run_all.sh             # all three approaches sequentially"
echo "  bash run_all.sh --debug     # quick 25-min validation run first"
