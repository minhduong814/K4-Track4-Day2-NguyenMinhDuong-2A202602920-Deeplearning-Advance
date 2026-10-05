"""Validation-selected inference methods for DeepWeeds classifiers."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_logits(model, loader, device, view=None):
    """Run one deterministic view over a loader and return names, labels, and logits."""
    device = torch.device(device)
    model.eval()
    filenames: list[str] = []
    labels, logits = [], []
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(images)
            filenames.extend(list(names))
            labels.append(target.cpu())
            logits.append(output.float().cpu())
    if not logits:
        raise ValueError("loader produced no batches")
    return filenames, torch.cat(labels).numpy(), torch.cat(logits).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    """Horizontally flip an NCHW batch."""
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    """Return top-left, top-right, bottom-left, bottom-right, and center crops."""
    if x.ndim != 4:
        raise ValueError("views_multicrop expects an NCHW tensor")
    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError(f"crop must be in 1..{min(height, width)}")
    bottom, right = height - crop, width - crop
    center_y, center_x = bottom // 2, right // 2
    boxes = [(0, 0), (0, right), (bottom, 0), (bottom, right),
             (center_y, center_x)]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in boxes]


def views_multiscale(x, sizes):
    """Bilinearly resize an NCHW batch to each requested square resolution."""
    if x.ndim != 4:
        raise ValueError("views_multiscale expects an NCHW tensor")
    sizes = [int(size) for size in sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("sizes must contain positive integers")
    return [F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=True) for size in sizes]


def _softmax_numpy(logits):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Aggregate view logits in probability or logit space and return probabilities."""
    if not logits_per_view:
        raise ValueError("At least one view is required")
    arrays = [np.asarray(item) for item in logits_per_view]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("All view logits must have the same (N, K) shape")
    space = space.lower()
    if space == "prob":
        probs = np.mean([_softmax_numpy(item) for item in arrays], axis=0)
    elif space == "logit":
        probs = _softmax_numpy(np.mean(arrays, axis=0))
    else:
        raise ValueError("space must be 'prob' or 'logit'")
    return probs / probs.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Average aligned probability arrays from multiple models."""
    if not list_of_probs:
        raise ValueError("At least one probability array is required")
    arrays = [np.asarray(item, dtype=np.float64) for item in list_of_probs]
    shape = arrays[0].shape
    if len(shape) != 2 or any(item.shape != shape for item in arrays):
        raise ValueError("All probability arrays must have the same (N, K) shape")
    if any(not np.isfinite(item).all() or (item < 0).any() for item in arrays):
        raise ValueError("Probabilities must be finite and non-negative")
    if any(not np.allclose(item.sum(1), 1.0, atol=1e-5) for item in arrays):
        raise ValueError("Each probability row must sum to one")
    probs = np.mean(arrays, axis=0)
    return probs / probs.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Fit one positive temperature on validation NLL using LBFGS over log(T)."""
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("Expected logits (N, K) and labels (N,)")
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())


def apply_temperature(logits, T: float):
    """Apply a positive scalar temperature and return normalized probabilities."""
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    return _softmax_numpy(np.asarray(logits) / float(T))


def fuse_conv_bn(model):
    """Return an eval-mode copy with adjacent Conv2d/BatchNorm2d modules fused."""
    fused = copy.deepcopy(model).eval()

    def recurse(parent: nn.Module) -> None:
        children = list(parent.named_children())
        for _, child in children:
            recurse(child)
        for index in range(len(children) - 1):
            conv_name, conv = children[index]
            bn_name, bn = children[index + 1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                setattr(parent, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, nn.Identity())

    recurse(fused)
    return fused
