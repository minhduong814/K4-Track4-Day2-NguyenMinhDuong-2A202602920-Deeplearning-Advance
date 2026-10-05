"""Model construction, freezing, optimizer groups, and model complexity."""
from __future__ import annotations

import torch
import torch.nn as nn

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _classifier_parameters(model) -> list[nn.Parameter]:
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        return list(classifier.parameters())
    if isinstance(classifier, str):
        module = model.get_submodule(classifier)
        return list(module.parameters())
    raise TypeError(f"Unsupported classifier returned by timm: {type(classifier)!r}")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Create a timm classifier using scratch, frozen, or full fine-tuning initialization."""
    import timm

    init = init.lower()
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be 'scratch', 'frozen', or 'finetune'")
    use_pretrained = False if init == "scratch" else bool(pretrained)
    model = timm.create_model(
        name,
        pretrained=use_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    if init == "frozen":
        freeze_backbone(model)

    cfg = getattr(model, "pretrained_cfg", {}) or {}
    source = cfg.get("hf_hub_id") or cfg.get("url") or cfg.get("architecture") or "none"
    model.weight_tag = str(source) if use_pretrained else "random_init"
    model.initialization = init
    return model


def freeze_backbone(model) -> None:
    """Freeze every parameter except those belonging to timm's classifier head."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    head = _classifier_parameters(model)
    if not head:
        raise ValueError("The model classifier has no trainable parameters")
    for parameter in head:
        parameter.requires_grad = True


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Return backbone decay/no-decay and head parameter groups for AdamW."""
    head_ids = {id(parameter) for parameter in _classifier_parameters(model)}
    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []
    head: list[nn.Parameter] = []
    seen: set[int] = set()

    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError("A model parameter was encountered more than once")
        seen.add(id(parameter))
        if id(parameter) in head_ids:
            head.append(parameter)
        elif parameter.ndim <= 1:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)

    groups = []
    if backbone_decay:
        groups.append({"params": backbone_decay, "lr": lr_backbone,
                       "weight_decay": weight_decay, "group_name": "backbone_decay"})
    if backbone_no_decay:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone,
                       "weight_decay": 0.0, "group_name": "backbone_no_decay"})
    if head:
        groups.append({"params": head, "lr": lr_head,
                       "weight_decay": weight_decay, "group_name": "head"})
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups


def count_params(model) -> float:
    """Return the total parameter count in millions, including frozen parameters."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000.0


_PROFILING_BUFFER_NAMES = {"total_ops", "total_params"}


def remove_profiling_buffers(model) -> int:
    """Remove temporary buffers registered by THOP and return how many were removed."""
    removed = 0
    for module in model.modules():
        for name in _PROFILING_BUFFER_NAMES:
            if name in module._buffers:
                module._buffers.pop(name)
                removed += 1
    return removed


def clean_profiling_state_dict(state_dict):
    """Return a state dict without THOP's non-model ``total_*`` buffers."""
    return type(state_dict)(
        (key, value) for key, value in state_dict.items()
        if key.rsplit(".", 1)[-1] not in _PROFILING_BUFFER_NAMES
    )


def count_gmacs(model, img_size: int = 224) -> float:
    """Count multiply-accumulates for one image with THOP and return GMAC."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("count_gmacs requires `pip install thop`") from exc

    try:
        parameter = next(model.parameters())
        device, dtype = parameter.device, parameter.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32
    was_training = model.training
    # THOP registers total_ops/total_params on the supplied model. Always clean
    # them so they never leak into training checkpoints.
    remove_profiling_buffers(model)
    model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size, device=device, dtype=dtype)
    try:
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
    finally:
        remove_profiling_buffers(model)
        model.train(was_training)
    return float(macs) / 1_000_000_000.0
