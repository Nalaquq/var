"""
dataset.py — MVFoul dataset classes and feature extraction utilities.
Shared by all three approaches (A, B, C).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# ── Class definitions (fixed by SoccerNet evaluation code) ────────────────────
ACTION_CLASSES = [
    "Tackling", "Standing tackling", "High leg", "Holding",
    "Pushing", "Elbowing", "Challenge", "Dive",
]
CLASS2IDX = {c: i for i, c in enumerate(ACTION_CLASSES)}
N_CLASSES  = len(ACTION_CLASSES)
DIVE_IDX   = CLASS2IDX["Dive"]

SEVERITY_MAP = {
    "1.0": "No card", "3.0": "Yellow card", "5.0": "Red card",
    "2.0": "No card/Yellow card", "4.0": "Yellow card/Red card", "": "Unknown",
}

# ── COCO-17 skeleton (subset of MediaPipe 33-kp) ──────────────────────────────
COCO_FROM_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
N_JOINTS     = 17
JOINT_NAMES  = [
    "Nose", "L-eye", "R-eye", "L-ear", "R-ear",
    "L-shoulder", "R-shoulder", "L-elbow", "R-elbow",
    "L-wrist", "R-wrist", "L-hip", "R-hip",
    "L-knee", "R-knee", "L-ankle", "R-ankle",
]
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_annotations(mvfoul_dir: Path, splits: list[str]) -> dict[str, "pd.DataFrame"]:
    """
    Load annotation JSONs for requested splits.
    Returns dict: split_name -> DataFrame with columns:
        action_id, label, action_name, is_dive, clip_paths, Severity
    """
    import pandas as pd

    split_dfs = {}
    for json_path in sorted(mvfoul_dir.rglob("*.json")):
        path_lower = str(json_path).lower()
        for s in splits:
            if s in path_lower and s not in split_dfs:
                split_dfs[s] = _parse_annotation_json(json_path, s)
                log.info("Loaded %s: %d actions from %s",
                         s, len(split_dfs[s]), json_path.name)
                break

    missing = [s for s in splits if s not in split_dfs]
    if missing:
        raise FileNotFoundError(
            f"Could not find annotation JSONs for splits: {missing}\n"
            f"Searched in: {mvfoul_dir}"
        )
    return split_dfs


def _parse_annotation_json(json_path: Path, split_name: str):
    import pandas as pd

    with open(json_path) as f:
        raw = json.load(f)
    actions = raw.get("Actions", raw.get("Set", {}))
    rows = []
    for action_id, data in actions.items():
        clips  = data.get("clips", [])
        action = data.get("Action class", "").strip()
        if action not in CLASS2IDX:
            action = "Dont know"
        rows.append({
            "action_id"  : action_id,
            "split"      : split_name,
            "label"      : CLASS2IDX.get(action, -1),
            "action_name": action,
            "is_dive"    : action == "Dive",
            "clip_paths" : [c.get("path", "") for c in clips],
            "Severity"   : SEVERITY_MAP.get(
                str(data.get("Severity", "")).strip(), "Unknown"
            ),
        })
    df = pd.DataFrame(rows)
    # Drop "Dont know" actions — excluded from official evaluation
    df = df[df["label"] >= 0].reset_index(drop=True)
    return df


def build_video_lookup(mvfoul_dir: Path) -> dict[str, list[Path]]:
    """Scan for video files and build action_id -> [clip_path, ...] mapping."""
    VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}
    lookup = defaultdict(list)
    for vp in mvfoul_dir.rglob("*"):
        if vp.suffix.lower() in VIDEO_EXTS:
            lookup[vp.parent.name].append(vp)
    return dict(lookup)


def compute_class_weights(df, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights, normalised so mean = 1."""
    counts  = df["label"].value_counts().sort_index()
    arr     = np.array([counts.get(i, 1) for i in range(N_CLASSES)], dtype=float)
    inv     = 1.0 / arr
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_adjacency(n_joints: int, edges: list[tuple]) -> torch.Tensor:
    """Normalised symmetric adjacency A_hat = D^{-1/2} A D^{-1/2}."""
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(n_joints, dtype=np.float32)
    D = np.diag(A.sum(axis=1) ** -0.5)
    return torch.tensor(D @ A @ D, dtype=torch.float32)


A_HAT = build_adjacency(N_JOINTS, SKELETON_EDGES)  # pre-built, shared


# ─────────────────────────────────────────────────────────────────────────────
# Video frame loading (Approach A)
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_frames(video_path: Path, n_frames: int = 16,
                     size: int = 224) -> torch.Tensor | None:
    """
    Uniformly sample n_frames from a video clip.
    Returns float tensor (3, T, H, W) normalised to ImageNet stats,
    or None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (size, size))
        frames.append(frame)
    cap.release()

    t    = torch.from_numpy(np.stack(frames)).float() / 255.0
    mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 1, 1, 3)
    std  = torch.tensor([0.225, 0.225, 0.225]).view(1, 1, 1, 3)
    return ((t - mean) / std).permute(3, 0, 1, 2)  # (3, T, H, W)


def _best_clip(clips: list[Path]) -> list[Path]:
    """Prefer replay clips (more informative per VARS paper)."""
    replays = [c for c in clips if "replay" in str(c).lower()]
    return replays or clips


class VideoDataset(Dataset):
    def __init__(self, df, video_lookup: dict, n_frames: int = 16, size: int = 224):
        self.df           = df.reset_index(drop=True)
        self.video_lookup = video_lookup
        self.n_frames     = n_frames
        self.size         = size
        self.synthetic    = not bool(video_lookup)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])

        if self.synthetic:
            return torch.randn(3, self.n_frames, self.size, self.size), label

        clips = _best_clip(self.video_lookup.get(row["action_id"], []))
        for vp in clips:
            t = load_clip_frames(vp, self.n_frames, self.size)
            if t is not None:
                return t, label

        return torch.randn(3, self.n_frames, self.size, self.size), label


# ─────────────────────────────────────────────────────────────────────────────
# Pose extraction (Approach B & C) with disk caching
# ─────────────────────────────────────────────────────────────────────────────

def extract_pose_sequence(video_path: Path,
                          n_frames: int = 16) -> np.ndarray | None:
    """
    Run MediaPipe Pose on n_frames sampled from the clip.
    Returns (n_frames, 33 * 3) float32 array [x, y, visibility] per keypoint,
    or None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    seq     = []

    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                seq.append(np.zeros(33 * 3, dtype=np.float32))
                continue
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                kps = [
                    [lm.x, lm.y, lm.visibility]
                    for lm in result.pose_landmarks.landmark
                ]
            else:
                kps = [[0.0, 0.0, 0.0]] * 33
            seq.append(np.array(kps, dtype=np.float32).flatten())
    cap.release()
    return np.array(seq, dtype=np.float32)


def get_pose_cached(action_id: str, clips: list[Path],
                    cache_dir: Path, n_frames: int = 16) -> np.ndarray:
    """Return cached pose sequence or extract and cache it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"pose_{action_id}.npy"

    if cache_file.exists():
        return np.load(cache_file)

    seq = None
    for vp in _best_clip(clips):
        seq = extract_pose_sequence(vp, n_frames)
        if seq is not None:
            break

    if seq is None:
        seq = np.zeros((n_frames, 33 * 3), dtype=np.float32)

    np.save(cache_file, seq)
    return seq


def pose_to_graph(seq: np.ndarray) -> np.ndarray:
    """
    Convert raw MediaPipe 33-kp sequence (T, 33*3) to velocity-augmented
    COCO-17 graph features (T, N_JOINTS, 4) = [x, y, vx, vy].
    Velocity is the contact-response signal at the heart of the project hypothesis.
    """
    T    = seq.shape[0]
    full = seq.reshape(T, 33, 3)
    coco = full[:, COCO_FROM_MP, :2]    # (T, 17, 2) — x, y only
    vel  = np.zeros_like(coco)
    vel[1:] = coco[1:] - coco[:-1]     # frame-over-frame displacement
    return np.concatenate([coco, vel], axis=-1).astype(np.float32)  # (T, 17, 4)


def get_graph_cached(action_id: str, clips: list[Path],
                     cache_dir: Path, n_frames: int = 16) -> np.ndarray:
    """Return cached graph features (T, 17, 4) or build from pose cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"graph_{action_id}.npy"

    if cache_file.exists():
        return np.load(cache_file)

    # Build from pose cache if it exists (avoids re-running MediaPipe)
    pose_file = cache_dir / f"pose_{action_id}.npy"
    if pose_file.exists():
        seq = np.load(pose_file)
    else:
        seq = None
        for vp in _best_clip(clips):
            seq = extract_pose_sequence(vp, n_frames)
            if seq is not None:
                break
        if seq is None:
            seq = np.zeros((n_frames, 33 * 3), dtype=np.float32)

    feats = pose_to_graph(seq)
    np.save(cache_file, feats)
    return feats


def pre_extract_features(df, video_lookup: dict, cache_dir: Path,
                         n_frames: int = 16, mode: str = "both") -> None:
    """
    Pre-extract pose and/or graph features for all actions.
    mode: 'pose' | 'graph' | 'both'
    Run this once before training Approaches B and C.
    """
    from tqdm import tqdm

    ids = df["action_id"].tolist()
    log.info("Pre-extracting %s features for %d actions...", mode, len(ids))

    already_pose  = {f.stem.replace("pose_", "")  for f in cache_dir.glob("pose_*.npy")}
    already_graph = {f.stem.replace("graph_", "") for f in cache_dir.glob("graph_*.npy")}

    for action_id in tqdm(ids, desc=f"Feature extraction ({mode})"):
        clips = video_lookup.get(action_id, [])
        if mode in ("pose", "both") and action_id not in already_pose:
            get_pose_cached(action_id, clips, cache_dir, n_frames)
        if mode in ("graph", "both") and action_id not in already_graph:
            get_graph_cached(action_id, clips, cache_dir, n_frames)


class PoseDataset(Dataset):
    def __init__(self, df, video_lookup: dict, cache_dir: Path, n_frames: int = 16):
        self.df           = df.reset_index(drop=True)
        self.video_lookup = video_lookup
        self.cache_dir    = cache_dir
        self.n_frames     = n_frames
        self.synthetic    = not bool(video_lookup)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])

        if self.synthetic:
            return torch.randn(self.n_frames, 33 * 3), label

        seq = get_pose_cached(
            row["action_id"],
            self.video_lookup.get(row["action_id"], []),
            self.cache_dir, self.n_frames,
        )
        return torch.from_numpy(seq), label


class GraphDataset(Dataset):
    def __init__(self, df, video_lookup: dict, cache_dir: Path, n_frames: int = 16):
        self.df           = df.reset_index(drop=True)
        self.video_lookup = video_lookup
        self.cache_dir    = cache_dir
        self.n_frames     = n_frames
        self.synthetic    = not bool(video_lookup)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = int(row["label"])

        if self.synthetic:
            feats = np.random.randn(self.n_frames, N_JOINTS, 4).astype(np.float32)
        else:
            feats = get_graph_cached(
                row["action_id"],
                self.video_lookup.get(row["action_id"], []),
                self.cache_dir, self.n_frames,
            )
        # (T, N, C) → (C, T, N) for Conv2d
        return torch.from_numpy(feats).permute(2, 0, 1), label
