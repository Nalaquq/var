#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# get_data.sh — Transfer MVFoul dataset from Colab/Drive to local machines
#
# FASTEST OPTION (recommended): re-download directly from SoccerNet servers
# ALTERNATIVE: copy from Google Drive if you already have clips there
#
# Run on the A4000 machine first (it will store the master copy).
# The 3070 machine can either:
#   (a) mount the A4000's copy via NFS/sshfs, or
#   (b) run the same download script — SoccerNet deduplicates automatically.
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Edit these ────────────────────────────────────────────────────────────────
SOCCERNET_PASSWORD="your_password_here"     # your soccer-net.org password
DATA_DIR="/data/mvfoul"                      # where to store the dataset locally
# ─────────────────────────────────────────────────────────────────────────────

echo "=== MVFoul dataset download ==="
echo "Target: $DATA_DIR"
echo ""

source .venv/bin/activate   # or: source activate mvfoul

mkdir -p "$DATA_DIR/mvfouls"
cd "$DATA_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# OPTION 1 (RECOMMENDED): Direct SoccerNet download
# Downloads annotations + 224p video clips for train/valid/test splits.
# Total size: ~50 GB. The downloader resumes if interrupted.
# ─────────────────────────────────────────────────────────────────────────────
python - <<PYEOF
from SoccerNet.Downloader import SoccerNetDownloader
from pathlib import Path

d = SoccerNetDownloader(LocalDirectory="$DATA_DIR")
d.password = "$SOCCERNET_PASSWORD"

print("Downloading annotations (train, valid, test)...")
d.downloadDataTask(
    task="mvfouls",
    split=["train", "valid", "test"],
    password="$SOCCERNET_PASSWORD",
    verbose=True,
)
print("Done. Extracting zip files...")

import zipfile
mvfoul_dir = Path("$DATA_DIR") / "mvfouls"
for zip_path in sorted(mvfoul_dir.glob("*.zip")):
    split_name = zip_path.stem.replace("_720p", "")
    out_dir    = mvfoul_dir / split_name
    if out_dir.exists():
        print(f"  {zip_path.name} already extracted")
        continue
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

print("\\nDownload and extraction complete.")
print(f"Dataset location: {mvfoul_dir}")
PYEOF

echo ""
echo "=== Verification ==="
python - <<PYEOF
from pathlib import Path

mvfoul_dir = Path("$DATA_DIR/mvfouls")
json_files  = list(mvfoul_dir.rglob("*.json"))
video_exts  = {".mp4", ".avi", ".mkv", ".mov"}
video_files = [p for p in mvfoul_dir.rglob("*") if p.suffix.lower() in video_exts]

print(f"JSON annotation files : {len(json_files)}")
print(f"Video clip files      : {len(video_files)}")
if len(video_files) > 0:
    total_gb = sum(p.stat().st_size for p in video_files) / 1e9
    print(f"Total video size      : {total_gb:.1f} GB")
    print("STATUS: ready for training")
else:
    print("STATUS: annotations only — re-check password or run again")
PYEOF


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 2: Copy from Google Drive (if clips already there from Colab)
#
# Step 1 — in Colab, zip and upload to Drive:
#   import shutil
#   shutil.make_archive('/content/drive/MyDrive/mvfoul_train', 'zip',
#                       '/content/mvfoul_data/mvfouls/train')
#
# Step 2 — install rclone locally and configure it with your Google account:
#   sudo apt install rclone
#   rclone config   # follow prompts, name the remote "gdrive"
#
# Step 3 — copy to local machine:
#   mkdir -p /data/mvfoul/mvfouls
#   rclone copy gdrive:mvfoul_train.zip /data/mvfoul/mvfouls/
#   rclone copy gdrive:mvfoul_valid.zip /data/mvfoul/mvfouls/
#   cd /data/mvfoul/mvfouls && unzip mvfoul_train.zip -d train/
#   cd /data/mvfoul/mvfouls && unzip mvfoul_valid.zip -d valid/
#
# This is slower than Option 1 but useful if you've already run Colab
# and want to avoid re-downloading from SoccerNet.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# OPTION 3: Share dataset between the two local machines via sshfs
#
# On the 3070 machine (assuming A4000 has IP 192.168.1.100):
#   sudo apt install sshfs
#   mkdir -p /data/mvfoul
#   sshfs user@192.168.1.100:/data/mvfoul /data/mvfoul -o ro,allow_other
#
# Both machines will then read from the same dataset.
# The cache_dir in config.yaml should be LOCAL to each machine
# (pose/graph extraction is CPU-bound and fast enough to run on each).
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "Next steps:"
echo "  1. Edit src/config.yaml and set data.root: $DATA_DIR"
echo "  2. On the A4000:  python src/train.py --approach A --config src/config.yaml"
echo "  3. On the 3070 :  python src/train.py --approach B --config src/config.yaml"
echo "  4. On the 3070 :  python src/train.py --approach C --config src/config.yaml"
echo "  5. Monitor       : tensorboard --logdir results/tensorboard --port 6006"
