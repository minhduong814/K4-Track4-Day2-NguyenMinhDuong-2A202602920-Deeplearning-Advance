"""Vòng huấn luyện dùng chung cho mọi thí nghiệm DeepWeeds (B, T và F).

Mọi cấu hình chạy qua một hàm ``run(cfg)``; đổi thí nghiệm bằng cách đổi ``Config``.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dataset as data_module
import losses as loss_module
import model as model_module


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def _eval_api():
    try:
        from eval import compute_metrics, save_predictions
        return compute_metrics, save_predictions
    except ImportError:
        for parent in Path(__file__).resolve().parents:
            if (parent / "eval.py").is_file():
                sys.path.insert(0, str(parent))
                from eval import compute_metrics, save_predictions
                return compute_metrics, save_predictions
        raise ImportError("Could not find the repository's eval.py")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, CUDA, and deterministic cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """Build AdamW with separate backbone decay/no-decay and classifier groups."""
    groups = model_module.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Build an iteration-wise linear-warmup then cosine-decay schedule."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = min(total_steps, int(round(cfg.warmup_epochs * steps_per_epoch)))

    def factor(step: int) -> float:
        current = step + 1
        if warmup_steps > 0 and current <= warmup_steps:
            return current / warmup_steps
        progress = (current - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class EMA:
    """Exponential moving average of floating state, with exact copied integer buffers."""

    def __init__(self, model, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        for name, target in self.module.state_dict().items():
            value = source[name].detach()
            if target.is_floating_point():
                target.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                target.copy_(value)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Train for one epoch and return sample-weighted loss and current learning rates."""
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    total_loss, total_items = 0.0, 0
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    for images, target, _ in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if cfg.mix:
            images, mixed_targets = loss_module.mix_batch(images, target, cfg.mix_alpha, cfg.mix)
        else:
            mixed_targets = None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss = (loss_module.mixed_loss(criterion, logits, mixed_targets)
                    if mixed_targets is not None else criterion(logits, target))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        batch_items = images.size(0)
        total_loss += float(loss.detach()) * batch_items
        total_items += batch_items
    if total_items == 0:
        raise ValueError("Training loader produced no samples")
    return {
        "train_loss": total_loss / total_items,
        "lr_backbone": float(optimizer.param_groups[0]["lr"]),
        "lr_head": float(optimizer.param_groups[-1]["lr"]),
    }


def evaluate(model, loader, criterion, device):
    """Return filenames, labels, logits, and sample-weighted mean loss."""
    model.eval()
    filenames: list[str] = []
    all_target, all_logits = [], []
    total_loss, total_items = 0.0, 0
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, target)
            filenames.extend(list(names))
            all_target.append(target.cpu())
            all_logits.append(logits.float().cpu())
            total_loss += float(loss) * images.size(0)
            total_items += images.size(0)
    if total_items == 0:
        raise ValueError("Evaluation loader produced no samples")
    return (filenames, torch.cat(all_target).numpy(), torch.cat(all_logits).numpy(),
            total_loss / total_items)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Plot train/validation loss, validation metrics, and epoch-end LR."""
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("history is empty")
    frame = pd.DataFrame(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(frame["epoch"], frame["train_loss"], marker="o", label="train")
    axes[0].plot(frame["epoch"], frame["val_loss"], marker="o", label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], marker="o", label="macro-F1")
    axes[1].plot(frame["epoch"], frame["val_top1"], marker="o", label="top-1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics", ylim=(0, 1))
    axes[1].legend()
    axes[2].plot(frame["epoch"], frame["lr_backbone"], marker="o", label="backbone")
    axes[2].plot(frame["epoch"], frame["lr_head"], marker="o", label="head")
    axes[2].set(xlabel="Epoch", ylabel="Learning rate", title="Epoch-end LR")
    axes[2].set_yscale("log")
    axes[2].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(1, keepdims=True)


def run(cfg: Config) -> dict:
    """Train, select only on validation macro-F1, and optionally evaluate test once."""
    if cfg.epochs <= 0:
        raise ValueError("epochs must be positive")
    set_seed(cfg.seed)
    compute_metrics, save_predictions = _eval_api()
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_df, val_df, test_df = data_module.load_split(cfg.labels_dir, cfg.fold)
    split_report = data_module.check_split(train_df, val_df, test_df, cfg.images_dir)
    (output / "split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_transform = data_module.build_transforms(True, cfg.img_size, cfg.aug)
    eval_transform = data_module.build_transforms(False, cfg.img_size, cfg.aug)
    train_loader = data_module.make_loader(
        train_df, cfg.images_dir, train_transform, cfg.batch_size, True,
        cfg.sampler, cfg.num_workers,
    )
    val_loader = data_module.make_loader(
        val_df, cfg.images_dir, eval_transform, cfg.batch_size, False,
        None, cfg.num_workers,
    )
    test_loader = None
    if cfg.save_test_predictions:
        test_loader = data_module.make_loader(
            test_df, cfg.images_dir, eval_transform, cfg.batch_size, False,
            None, cfg.num_workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model_module.build_model(
        cfg.backbone, pretrained=cfg.init != "scratch", num_classes=data_module.NUM_CLASSES,
        drop_rate=cfg.drop_rate, init=cfg.init,
    ).to(device)
    params_m = model_module.count_params(network)
    gmacs = model_module.count_gmacs(network, cfg.img_size)

    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().reindex(
            range(data_module.NUM_CLASSES), fill_value=0).to_numpy()
        beta = 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta
        weight = loss_module.class_weights(counts, beta).to(device)
    criterion = loss_module.build_criterion(
        cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=weight,
        alpha=weight,
    ).to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None

    checkpoint_path = output / "best.pt"
    history_path = output / "history.csv"
    history: list[dict] = []
    best_f1, best_epoch = -math.inf, -1
    if checkpoint_path.is_file() and history_path.is_file():
        cached_history = pd.read_csv(history_path)
        if len(cached_history) >= cfg.epochs and int(cached_history["epoch"].max()) >= cfg.epochs:
            history = cached_history.iloc[:cfg.epochs].to_dict(orient="records")
            cached_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            best_f1 = float(cached_checkpoint["val_macro_f1"])
            best_epoch = int(cached_checkpoint["epoch"])
            print(f"[{cfg.exp_id} seed={cfg.seed}] reuse completed {cfg.epochs}-epoch run")

    for epoch in range(len(history) + 1, cfg.epochs + 1):
        start = time.perf_counter()
        train_stats = train_one_epoch(
            network, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema
        )
        eval_model = ema.module if ema is not None else network
        _, val_target, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = _probabilities(val_logits)
        metrics = compute_metrics(val_target, val_probs.argmax(1), val_probs)
        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_top1": metrics["top1"],
            "val_ece": metrics["ece"],
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = metrics["macro_f1"], epoch
            torch.save({
                "model": model_module.clean_profiling_state_dict(eval_model.state_dict()),
                "epoch": epoch,
                "val_macro_f1": best_f1,
                "config": asdict(cfg),
            }, checkpoint_path)
        print(f"[{cfg.exp_id} seed={cfg.seed}] {epoch:02d}/{cfg.epochs} "
              f"loss={row['train_loss']:.4f}/{val_loss:.4f} "
              f"val_f1={metrics['macro_f1']:.4f} val_top1={metrics['top1']:.4f}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_module.remove_profiling_buffers(network)
    clean_state = model_module.clean_profiling_state_dict(checkpoint["model"])
    network.load_state_dict(clean_state)
    val_names, val_target, val_logits, val_loss = evaluate(network, val_loader, criterion, device)
    val_probs = _probabilities(val_logits)
    val_metrics = compute_metrics(val_target, val_probs.argmax(1), val_probs)
    np.save(output / "val_logits.npy", val_logits)
    save_predictions(pred_path(cfg, "val"), val_names, val_target, val_probs)

    test_metrics = None
    if cfg.save_test_predictions:
        test_names, test_target, test_logits, _ = evaluate(
            network, test_loader, criterion, device
        )
        test_probs = _probabilities(test_logits)
        test_metrics = compute_metrics(test_target, test_probs.argmax(1), test_probs)
        np.save(output / "test_logits.npy", test_logits)
        save_predictions(pred_path(cfg, "test"), test_names, test_target, test_probs)

    curves_dir = Path(cfg.out_dir).parent / "curves"
    curve_path = curves_dir / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    summary = {
        "exp_id": cfg.exp_id,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "weight_tag": getattr(network, "weight_tag", "see config/checkpoint"),
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top1": val_metrics["top1"],
        "val_ece": val_metrics["ece"],
        "params_m": params_m,
        "gmacs": gmacs,
        "seconds_per_epoch": float(np.mean([row["seconds"] for row in history])),
        "curve": str(Path("curves") / curve_path.name),
        "test_macro_f1": None if test_metrics is None else test_metrics["macro_f1"],
        "test_top1": None if test_metrics is None else test_metrics["top1"],
        "test_ece": None if test_metrics is None else test_metrics["ece"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Parse and type-check command-line ``KEY=VALUE`` Config overrides."""
    definitions = {field.name: field for field in fields(Config)}
    defaults = Config()
    optional_float = {"class_weight_beta", "ema_decay"}
    optional_string = {"sampler", "mix"}
    parsed = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be KEY=VALUE, got '{pair}'")
        key, raw = pair.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in definitions:
            raise KeyError(f"Unknown Config field '{key}'")
        if raw.lower() in {"none", "null"}:
            if key not in optional_float | optional_string:
                raise ValueError(f"Config field '{key}' cannot be None")
            value = None
        elif key in optional_float:
            value = float(raw)
        elif key in optional_string:
            value = raw
        else:
            default = getattr(defaults, key)
            if isinstance(default, bool):
                if raw.lower() not in {"true", "false", "1", "0", "yes", "no"}:
                    raise ValueError(f"'{raw}' is not a boolean for {key}")
                value = raw.lower() in {"true", "1", "yes"}
            elif isinstance(default, int):
                value = int(raw)
            elif isinstance(default, float):
                value = float(raw)
            else:
                value = raw
        parsed[key] = value
    return parsed


def main() -> None:
    """CLI entry point: ``python train.py --set exp_id=B01 backbone=resnet50``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override Config fields")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
