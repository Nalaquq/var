"""
models.py — All three MVFoul model architectures.
  A: MViTv2-S video baseline (replicates VARS, Held et al. 2023)
  B: BiLSTM pose baseline    (replicates Fang et al. 2024)
  C: ST-GCN novel approach   (velocity-augmented skeleton graphs)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import N_CLASSES, N_JOINTS, A_HAT


# ─────────────────────────────────────────────────────────────────────────────
# Approach A — MViTv2-S Video Baseline
# ─────────────────────────────────────────────────────────────────────────────

class ApproachA_MViT(nn.Module):
    """
    MViTv2-S pretrained on Kinetics-400, fine-tuned for MVFoul classification.
    Two-phase training: freeze backbone first, then unfreeze for full fine-tune.
    Replicates the VARS paper (Held et al., CVPR Workshop 2023).

    Input:  (B, 3, T, H, W)
    Output: (B, n_classes)
    """

    def __init__(self, n_classes: int = N_CLASSES, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo", "mvit_v2_s", pretrained=True,
            verbose=False,
        )
        in_features = self.backbone.head.proj.in_features
        self.backbone.head.proj = nn.Linear(in_features, n_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "head" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Approach B — BiLSTM Pose Baseline
# ─────────────────────────────────────────────────────────────────────────────

class ApproachB_BiLSTM(nn.Module):
    """
    Bidirectional LSTM over MediaPipe 33-keypoint pose sequences.
    Replicates Fang, Yeung & Fujii (Sports Engineering, 2024).

    Input:  (B, T, 33 * 3)  — flattened [x, y, visibility] per keypoint
    Output: (B, n_classes)
    """

    def __init__(
        self,
        in_features: int = 33 * 3,
        hidden: int = 256,
        n_layers: int = 2,
        n_classes: int = N_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.lstm = nn.LSTM(
            hidden, hidden, n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = F.relu(self.proj(x))        # (B, T, hidden)
        out, _ = self.lstm(x)            # (B, T, 2 * hidden)
        pooled = out.mean(dim=1)         # (B, 2 * hidden)
        return self.head(pooled)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Approach C — Spatiotemporal Graph Convolutional Network (Novel)
# ─────────────────────────────────────────────────────────────────────────────

class SpatialGCN(nn.Module):
    """
    One spatial graph convolution: H' = relu(BN(A_hat @ H @ W))
    Input/Output: (B, C, T, N)
    """

    def __init__(self, c_in: int, c_out: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)        # (N, N) fixed normalised adjacency
        self.W  = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, N = x.shape
        # Reshape to apply linear across node features
        h = x.permute(0, 2, 3, 1).reshape(B * T, N, C)   # (BT, N, C)
        h = self.W(h)                                       # (BT, N, C_out)
        # Graph convolution: A_hat @ H
        h = torch.bmm(self.A.unsqueeze(0).expand(B * T, -1, -1), h)  # (BT, N, C_out)
        h = h.reshape(B, T, N, -1).permute(0, 3, 1, 2)   # (B, C_out, T, N)
        return F.relu(self.bn(h))


class STGCNBlock(nn.Module):
    """
    One ST-GCN block: spatial GCN → temporal Conv1D → residual add.
    Reference: Yan et al. (2018) 'Spatial Temporal Graph Convolutional Networks
    for Skeleton-Based Action Recognition', AAAI.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        A: torch.Tensor,
        t_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn = SpatialGCN(c_in, c_out, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                c_out, c_out,
                kernel_size=(t_kernel, 1),
                stride=(stride, 1),
                padding=(t_kernel // 2, 0),
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        # Residual projection when channel sizes differ
        if c_in != c_out or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x))


class ApproachC_STGCN(nn.Module):
    """
    Spatiotemporal Graph Convolutional Network for skeleton-based foul detection.

    Novel contribution: node features include velocity [vx, vy] in addition to
    position [x, y], encoding the contact-response signature that distinguishes
    genuine falls from simulated dives.

    Input:  (B, 4, T, N_JOINTS)  — [x, y, vx, vy] × T frames × 17 joints
    Output: (B, n_classes)

    Architecture: 3 ST-GCN blocks (64 → 128 → 256 channels) → global avg pool
    """

    def __init__(
        self,
        c_in: int = 4,
        n_classes: int = N_CLASSES,
        channels: list[int] = None,
        t_kernel: int = 9,
        dropout: float = 0.2,
        A: torch.Tensor = None,
    ):
        super().__init__()
        channels = channels or [64, 128, 256]
        A        = A if A is not None else A_HAT

        # Input batch norm over flattened node features
        self.bn_in = nn.BatchNorm1d(c_in * N_JOINTS)

        # Build ST-GCN layers from channel progression
        layer_defs = [(c_in, channels[0])] + list(zip(channels[:-1], channels[1:]))
        self.layers = nn.ModuleList([
            STGCNBlock(cin, cout, A, t_kernel=t_kernel, dropout=dropout)
            for cin, cout in layer_defs
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # global avg over (T, N)
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, N)
        B, C, T, N = x.shape
        # Input normalisation
        xn = x.permute(0, 2, 1, 3).reshape(B * T, C * N)
        xn = self.bn_in(xn).reshape(B, T, C, N).permute(0, 2, 1, 3)
        for layer in self.layers:
            xn = layer(xn)
        return self.head(xn)

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(approach: str, cfg: dict, device: torch.device) -> nn.Module:
    """Instantiate the requested model from config."""
    if approach == "A":
        model = ApproachA_MViT(n_classes=N_CLASSES, freeze_backbone=True)
    elif approach == "B":
        c = cfg["approach_B"]
        model = ApproachB_BiLSTM(
            hidden=c["hidden_dim"],
            n_layers=c["n_layers"],
            n_classes=N_CLASSES,
            dropout=c["dropout"],
        )
    elif approach == "C":
        c = cfg["approach_C"]
        model = ApproachC_STGCN(
            c_in=c["node_features"],
            n_classes=N_CLASSES,
            channels=c["channels"],
            t_kernel=c["t_kernel"],
            dropout=c["dropout"],
            A=A_HAT.to(device),
        )
    else:
        raise ValueError(f"Unknown approach: {approach!r}. Choose A, B, or C.")

    return model.to(device)
