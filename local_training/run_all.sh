#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh — Train all three approaches sequentially on the A4000.
# Expected total wall time: ~11 hours on RTX A4000 (full dataset).
# Debug run (60-action subset): ~25 min.
#
# Usage:
#   bash run_all.sh              # full training run
#   bash run_all.sh --debug      # quick pipeline validation (~25 min)
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/src/config.yaml"
DEBUG_FLAG=""

if [[ "$1" == "--debug" ]]; then
    DEBUG_FLAG="--debug"
    echo "=== DEBUG MODE — 60-action subset per split ==="
fi

echo "=== MVFoul — Full training run on RTX A4000 ==="
echo "Config: $CONFIG"
echo "Log:    results/run_all.log"
echo ""

START_TOTAL=$(date +%s)

# ── Approach B first (pre-extracts pose cache used by C too) ─────────────────
echo "[$(date '+%H:%M')] Starting Approach B (BiLSTM)..."
START=$(date +%s)
python "$SCRIPT_DIR/src/train.py" \
    --approach B \
    --config "$CONFIG" \
    $DEBUG_FLAG \
    2>&1 | tee -a results/run_all.log
echo "[$(date '+%H:%M')] Approach B done ($(( ($(date +%s) - START) / 60 )) min)"
echo ""

# ── Approach C (reuses pose cache from B, only adds velocity features) ────────
echo "[$(date '+%H:%M')] Starting Approach C (ST-GCN)..."
START=$(date +%s)
python "$SCRIPT_DIR/src/train.py" \
    --approach C \
    --config "$CONFIG" \
    --no_extract \
    $DEBUG_FLAG \
    2>&1 | tee -a results/run_all.log
echo "[$(date '+%H:%M')] Approach C done ($(( ($(date +%s) - START) / 60 )) min)"
echo ""

# ── Approach A last (heaviest, monopolises VRAM — run when you can step away) ─
echo "[$(date '+%H:%M')] Starting Approach A (MViT)..."
START=$(date +%s)
python "$SCRIPT_DIR/src/train.py" \
    --approach A \
    --config "$CONFIG" \
    $DEBUG_FLAG \
    2>&1 | tee -a results/run_all.log
echo "[$(date '+%H:%M')] Approach A done ($(( ($(date +%s) - START) / 60 )) min)"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$(( ($(date +%s) - START_TOTAL) / 60 ))
echo "=== All approaches complete (total: ${TOTAL} min) ==="
echo ""
echo "Results:"
python - <<'PYEOF'
import json
from pathlib import Path

results_dir = Path("results")
approaches  = [("A", "MViT video"), ("B", "BiLSTM pose"), ("C", "ST-GCN novel")]

print(f"{'Approach':<22} {'Balanced Acc':>14} {'Dive F1':>10} {'Macro F1':>10}")
print("-" * 60)
for code, name in approaches:
    p = results_dir / f"approach_{code}_metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        print(f"{code} — {name:<18} {m['balanced_acc']:>14.4f} {m['dive_f1']:>10.4f} {m['macro_f1']:>10.4f}")
    else:
        print(f"{code} — {name:<18} {'(not found)':>14}")

print()
print(f"Checkpoints : checkpoints/")
print(f"TensorBoard : tensorboard --logdir results/tensorboard")
PYEOF
