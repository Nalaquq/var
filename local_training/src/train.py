"""
train.py — Main training entry point for all three MVFoul approaches.

Usage examples:
  # A4000 machine — run Approach A (MViT, needs 16 GB VRAM)
  python src/train.py --approach A --config src/config.yaml

  # 3070 machine — run Approach B then C
  python src/train.py --approach B --config src/config.yaml
  python src/train.py --approach C --config src/config.yaml

  # Quick smoke test on any machine
  python src/train.py --approach C --config src/config.yaml --debug

  # Override a config value without editing the YAML
  python src/train.py --approach A --config src/config.yaml --batch_size 2
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    ACTION_CLASSES,
    DIVE_IDX,
    N_CLASSES,
    N_JOINTS,
    JOINT_NAMES,
    GraphDataset,
    PoseDataset,
    VideoDataset,
    build_video_lookup,
    compute_class_weights,
    load_annotations,
    pre_extract_features,
)
from models import ApproachA_MViT, build_model

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MVFoul training script")
    p.add_argument("--approach",    required=True, choices=["A", "B", "C"],
                   help="Which approach to train")
    p.add_argument("--config",      required=True,
                   help="Path to config.yaml")
    p.add_argument("--data_dir",    default=None,
                   help="Override data.root from config")
    p.add_argument("--results_dir", default=None,
                   help="Override data.results_dir from config")
    p.add_argument("--debug",       action="store_true",
                   help="Override debug_mode=True (fast pipeline validation)")
    p.add_argument("--batch_size",  type=int, default=None,
                   help="Override batch_size for the selected approach")
    p.add_argument("--gpu",         type=int, default=0,
                   help="CUDA device index (default: 0)")
    p.add_argument("--no_extract",  action="store_true",
                   help="Skip pre-extraction of pose/graph features")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    ba      = balanced_accuracy_score(y_true, y_pred)
    macro   = f1_score(y_true, y_pred, average="macro",
                       labels=list(range(N_CLASSES)), zero_division=0)
    dive_f1 = f1_score(y_true, y_pred, labels=[DIVE_IDX],
                       average="macro", zero_division=0)
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    return {
        f"{prefix}balanced_acc": round(ba, 4),
        f"{prefix}dive_f1"     : round(dive_f1, 4),
        f"{prefix}macro_f1"    : round(macro, 4),
        f"{prefix}conf_matrix" : cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total_loss, preds, trues = 0.0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss   = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        preds.extend(logits.detach().argmax(1).cpu().numpy())
        trues.extend(y.cpu().numpy())

    n = len(trues)
    return total_loss / n, balanced_accuracy_score(trues, preds)


@torch.no_grad()
def run_eval(model, loader, device) -> dict:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast():
            preds.extend(model(x).argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return compute_metrics(trues, preds)


def train_phase(
    name: str, model, train_dl, valid_dl, device,
    n_epochs: int, lr: float, weight_decay: float,
    patience: int, weights: torch.Tensor,
    ckpt_path: Path, writer: SummaryWriter, epoch_offset: int = 0,
):
    """Run one training phase. Returns (best_metrics, history_df)."""
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler    = GradScaler()

    best_ba, best_metrics, patience_ctr = 0.0, {}, 0
    history = []

    log.info("Phase: %s | epochs=%d | lr=%.1e | patience=%d", name, n_epochs, lr, patience)

    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        tr_loss, tr_ba = train_one_epoch(model, train_dl, optimizer, scaler, criterion, device)
        vm             = run_eval(model, valid_dl, device)
        scheduler.step()

        global_ep = epoch_offset + ep
        writer.add_scalar("train/loss",      tr_loss,               global_ep)
        writer.add_scalar("train/balanced_acc", tr_ba,              global_ep)
        writer.add_scalar("valid/balanced_acc", vm["balanced_acc"], global_ep)
        writer.add_scalar("valid/dive_f1",      vm["dive_f1"],      global_ep)

        row = {"epoch": global_ep, "phase": name,
               "tr_loss": tr_loss, "tr_ba": tr_ba,
               **{k: v for k, v in vm.items() if k != "conf_matrix"}}
        history.append(row)

        flag = ""
        if vm["balanced_acc"] > best_ba:
            best_ba      = vm["balanced_acc"]
            best_metrics = vm
            torch.save(model.state_dict(), ckpt_path)
            patience_ctr = 0
            flag = " ✓"
        else:
            patience_ctr += 1

        log.info(
            "ep %02d/%02d  loss=%.3f  tr_ba=%.3f  val_ba=%.3f  dive_f1=%.3f  (%.1fs)%s",
            ep, n_epochs, tr_loss, tr_ba,
            vm["balanced_acc"], vm["dive_f1"],
            time.time() - t0, flag,
        )

        if patience_ctr >= patience:
            log.info("Early stop at epoch %d", ep)
            break

    # Restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    log.info("Phase %s done | best val_ba=%.4f  dive_f1=%.4f",
             name, best_ba, best_metrics.get("dive_f1", 0))
    return best_metrics, pd.DataFrame(history)


# ─────────────────────────────────────────────────────────────────────────────
# Joint importance ablation (Approach C only)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def joint_importance_ablation(model, valid_dl, device) -> pd.DataFrame:
    """
    Zero out one joint at a time and measure Dive F1 drop.
    The joints with the largest drop are most discriminative for Dive detection.
    """
    model.eval()
    rows = []

    # Baseline Dive F1
    preds, trues = [], []
    for x, y in valid_dl:
        x = x.to(device)
        with autocast():
            preds.extend(model(x).argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    baseline_f1 = f1_score(trues, preds, labels=[DIVE_IDX],
                            average="macro", zero_division=0)

    for j, jname in enumerate(JOINT_NAMES):
        preds_j = []
        for x, y in valid_dl:
            xc = x.clone().to(device)
            xc[:, :, :, j] = 0.0  # zero all channels for joint j
            with autocast():
                preds_j.extend(model(xc).argmax(1).cpu().numpy())
        f1_j = f1_score(trues, preds_j, labels=[DIVE_IDX],
                         average="macro", zero_division=0)
        drop = baseline_f1 - f1_j
        rows.append({"joint": jname, "joint_idx": j,
                     "dive_f1_zeroed": round(f1_j, 4),
                     "dive_f1_drop": round(drop, 4)})
        log.info("  %-14s  drop: %+.4f", jname, drop)

    return pd.DataFrame(rows).sort_values("dive_f1_drop", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Debug subset helper
# ─────────────────────────────────────────────────────────────────────────────

def apply_debug_subset(split_dfs: dict, n: int, seed: int) -> dict:
    import pandas as pd
    out = {}
    for s, df in split_dfs.items():
        out[s] = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(
                  min(len(x), max(1, n // N_CLASSES)), random_state=seed))
              .reset_index(drop=True)
        )
        log.info("Debug subset [%s]: %d → %d actions (dive=%d)",
                 s, len(df), len(out[s]), out[s]["is_dive"].sum())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["training"]
    seed      = train_cfg["seed"]
    debug     = args.debug or train_cfg.get("debug_mode", False)
    debug_n   = train_cfg.get("debug_n", 60)

    data_root   = Path(args.data_dir or cfg["data"]["root"])
    results_dir = Path(args.results_dir or cfg["data"]["results_dir"])
    ckpt_dir    = Path(cfg["data"]["ckpt_dir"])
    cache_dir   = Path(cfg["data"]["cache_dir"])
    mvfoul_dir  = data_root / "mvfouls"

    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file = results_dir / f"train_approach_{args.approach}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    log.info("=== MVFoul Training — Approach %s ===", args.approach)
    log.info("Config : %s", args.config)
    log.info("Data   : %s", mvfoul_dir)
    log.info("Debug  : %s", debug)

    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info("Device : %s", device)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.gpu)
        log.info("GPU    : %s  VRAM: %.1f GB", props.name, props.total_memory / 1e9)

    # ── Data ──────────────────────────────────────────────────────────────────
    splits    = [cfg["data"]["splits"][s] for s in ("train", "valid")]
    split_dfs = load_annotations(mvfoul_dir, splits)

    train_df = split_dfs[cfg["data"]["splits"]["train"]]
    valid_df = split_dfs[cfg["data"]["splits"]["valid"]]

    if debug:
        split_dfs = apply_debug_subset(
            {cfg["data"]["splits"]["train"]: train_df,
             cfg["data"]["splits"]["valid"]: valid_df},
            debug_n, seed,
        )
        train_df = split_dfs[cfg["data"]["splits"]["train"]]
        valid_df = split_dfs[cfg["data"]["splits"]["valid"]]

    log.info("Train: %d actions (dive=%d)", len(train_df), train_df["is_dive"].sum())
    log.info("Valid: %d actions (dive=%d)", len(valid_df), valid_df["is_dive"].sum())

    video_lookup = build_video_lookup(mvfoul_dir)
    has_videos   = bool(video_lookup)
    if not has_videos:
        log.warning("No video files found — running in SYNTHETIC mode (random tensors).")

    weights = compute_class_weights(train_df, device)
    log.info("Dive class weight: %.2f", weights[DIVE_IDX].item())

    num_workers = train_cfg.get("num_workers", 4)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_dir = results_dir / f"tensorboard/approach_{args.approach}"
    writer = SummaryWriter(log_dir=str(tb_dir))
    log.info("TensorBoard: tensorboard --logdir %s", tb_dir)

    # ── Approach-specific setup ───────────────────────────────────────────────
    ckpt_path = ckpt_dir / f"approach_{args.approach}_best.pt"
    model     = build_model(args.approach, cfg, device)
    log.info("Model params: %s", getattr(model, "trainable_params", "N/A"))

    approach_cfg = cfg[f"approach_{args.approach}"]
    n_frames     = approach_cfg.get("n_frames", 16)
    batch_size   = args.batch_size or approach_cfg["batch_size"]

    if args.approach == "A":
        # ── Approach A ──────────────────────────────────────────────────────
        frame_size = approach_cfg.get("frame_size", 224)
        train_ds   = VideoDataset(train_df, video_lookup, n_frames, frame_size)
        valid_ds   = VideoDataset(valid_df, video_lookup, n_frames, frame_size)
        train_dl   = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=False  # WSL2: persistent workers cause hangs)
        valid_dl   = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=False  # WSL2: persistent workers cause hangs)

        # Phase 1: head only
        p1 = approach_cfg["phase1"]
        final_metrics, hist1 = train_phase(
            "A-head-only", model, train_dl, valid_dl, device,
            n_epochs=p1["n_epochs"], lr=p1["lr"],
            weight_decay=approach_cfg["weight_decay"],
            patience=p1["patience"], weights=weights,
            ckpt_path=ckpt_path, writer=writer, epoch_offset=0,
        )

        # Phase 2: full fine-tune
        model.unfreeze_backbone()
        p2 = approach_cfg["phase2"]
        final_metrics, hist2 = train_phase(
            "A-full-finetune", model, train_dl, valid_dl, device,
            n_epochs=p2["n_epochs"], lr=p2["lr"],
            weight_decay=approach_cfg["weight_decay"],
            patience=p2["patience"], weights=weights,
            ckpt_path=ckpt_path, writer=writer,
            epoch_offset=p1["n_epochs"],
        )
        history = pd.concat([hist1, hist2], ignore_index=True)
        imp_df   = None

    elif args.approach == "B":
        # ── Approach B ──────────────────────────────────────────────────────
        if has_videos and not args.no_extract:
            log.info("Pre-extracting pose features...")
            all_df = pd.concat([train_df, valid_df], ignore_index=True)
            pre_extract_features(all_df, video_lookup, cache_dir, n_frames, mode="pose")

        train_ds = PoseDataset(train_df, video_lookup, cache_dir, n_frames)
        valid_ds = PoseDataset(valid_df, video_lookup, cache_dir, n_frames)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False  # WSL2: persistent workers cause hangs)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False  # WSL2: persistent workers cause hangs)

        final_metrics, history = train_phase(
            "B-BiLSTM", model, train_dl, valid_dl, device,
            n_epochs=approach_cfg["n_epochs"], lr=approach_cfg["lr"],
            weight_decay=approach_cfg["weight_decay"],
            patience=approach_cfg["patience"], weights=weights,
            ckpt_path=ckpt_path, writer=writer,
        )
        imp_df = None

    elif args.approach == "C":
        # ── Approach C ──────────────────────────────────────────────────────
        if has_videos and not args.no_extract:
            log.info("Pre-extracting graph features...")
            all_df = pd.concat([train_df, valid_df], ignore_index=True)
            pre_extract_features(all_df, video_lookup, cache_dir, n_frames, mode="both")

        train_ds = GraphDataset(train_df, video_lookup, cache_dir, n_frames)
        valid_ds = GraphDataset(valid_df, video_lookup, cache_dir, n_frames)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False  # WSL2: persistent workers cause hangs)
        valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=False  # WSL2: persistent workers cause hangs)

        final_metrics, history = train_phase(
            "C-STGCN", model, train_dl, valid_dl, device,
            n_epochs=approach_cfg["n_epochs"], lr=approach_cfg["lr"],
            weight_decay=approach_cfg["weight_decay"],
            patience=approach_cfg["patience"], weights=weights,
            ckpt_path=ckpt_path, writer=writer,
        )

        # Joint importance ablation
        log.info("Running joint importance ablation...")
        imp_df = joint_importance_ablation(model, valid_dl, device)
        imp_path = results_dir / "approach_C_joint_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        log.info("Saved joint importance: %s", imp_path)
        log.info("Top 5 most diagnostic joints:\n%s",
                 imp_df.head(5).to_string(index=False))

    # ── Save results ──────────────────────────────────────────────────────────
    hist_path = results_dir / f"approach_{args.approach}_history.csv"
    history.to_csv(hist_path, index=False)
    log.info("History saved: %s", hist_path)

    results_out = {k: (v.tolist() if hasattr(v, "tolist") else float(v))
                   for k, v in final_metrics.items()}
    json_path = results_dir / f"approach_{args.approach}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_out, f, indent=2)
    log.info("Metrics saved: %s", json_path)

    writer.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"APPROACH {args.approach} — FINAL VALIDATION METRICS")
    print("=" * 60)
    print(f"  Balanced Accuracy : {final_metrics['balanced_acc']:.4f}")
    print(f"  Dive F1           : {final_metrics['dive_f1']:.4f}")
    print(f"  Macro F1          : {final_metrics['macro_f1']:.4f}")
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"  TensorBoard       : tensorboard --logdir {tb_dir}")
    if not has_videos:
        print("\n  ⚠  SYNTHETIC MODE — metrics are not meaningful.")
        print("     Download video clips and re-run for real results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
