"""Classification losses and batch-level Mixup/CutMix."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Build CE, label-smoothed CE, focal loss, or class-weighted CE."""
    kind = kind.lower()
    weight = kw.get("weight")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", 0.1)), weight=weight)
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), alpha=kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires a class-weight tensor")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss '{kind}'; use ce/ls/focal/ce_weighted")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy using PyTorch's epsilon/K label-smoothing convention."""

    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight,
                               label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multi-class focal loss; gamma=0 is exactly weighted/unweighted CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim != 1:
                raise ValueError("alpha must be a one-dimensional class-weight vector")
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.gather(0, target)
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Compute inverse-frequency or effective-number weights with mean one."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or len(counts) != 9:
        raise ValueError("counts must contain exactly 9 class counts")
    if (counts <= 0).any():
        raise ValueError("every training class count must be positive")
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1)")
    if beta == 0:
        weights = counts.reciprocal()
    else:
        beta_tensor = torch.tensor(beta, dtype=torch.float64)
        weights = (1.0 - beta_tensor) / (1.0 - torch.pow(beta_tensor, counts))
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Apply Mixup or CutMix and return mixed images plus paired hard targets."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    mode = mode.lower()
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be 'mixup' or 'cutmix'")
    if len(x) != len(y):
        raise ValueError("x and y must have the same batch size")
    if len(x) < 2 or alpha == 0:
        return x, (y, y, 1.0)

    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    y_b = y[perm]
    if mode == "mixup":
        mixed = x * lam + x[perm] * (1.0 - lam)
        return mixed, (y, y_b, lam)

    if x.ndim != 4:
        raise ValueError("CutMix expects an NCHW image batch")
    height, width = x.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = int(torch.randint(0, width, (1,), device=x.device).item())
    center_y = int(torch.randint(0, height, (1,), device=x.device).item())
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + (cut_w + 1) // 2, width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + (cut_h + 1) // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y, y_b, actual_lam)


def mixed_loss(criterion, logits, targets):
    """Combine criterion values for Mixup/CutMix target pairs."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
