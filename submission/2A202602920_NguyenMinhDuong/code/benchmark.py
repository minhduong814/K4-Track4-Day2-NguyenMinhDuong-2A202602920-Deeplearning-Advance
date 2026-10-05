"""Synchronized and percentile-based inference latency measurements."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Measure a no-argument callable and return latency percentiles in milliseconds."""
    if warmup < 10:
        raise ValueError("warmup must be at least 10")
    if iters < 50:
        raise ValueError("iters must be at least 50")
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    samples = np.empty(iters, dtype=np.float64)
    for index in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples[index] = (time.perf_counter() - start) * 1_000.0
    return {
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "mean": float(samples.mean()),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Benchmark model forward only; preprocessing and host/device transfer are excluded."""
    if batch_size <= 0 or img_size <= 0:
        raise ValueError("batch_size and img_size must be positive")
    dtype = dtype.lower()
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_obj.type != "cuda" and dtype == "fp16":
        raise ValueError("fp16 benchmarking is only supported on CUDA")

    model = model.to(device_obj).eval()
    try:
        original_dtype = next(model.parameters()).dtype
    except StopIteration:
        original_dtype = torch.float32
    input_dtype = torch.float16 if dtype == "fp16" else torch.float32
    if dtype == "fp16":
        model.half()
    else:
        model.float()
    sample = torch.randn(batch_size, 3, img_size, img_size,
                         device=device_obj, dtype=input_dtype)

    def forward():
        with torch.inference_mode(), torch.autocast(
                device_type=device_obj.type,
                enabled=dtype == "amp",
                dtype=torch.float16 if device_obj.type == "cuda" else torch.bfloat16):
            model(sample)

    sync = torch.cuda.synchronize if device_obj.type == "cuda" else None
    try:
        stats = bench(forward, warmup=warmup, iters=iters, sync=sync)
    finally:
        model.to(dtype=original_dtype)
    stats.update({
        "gpu": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU",
        "dtype": dtype,
        "batch": int(batch_size),
        "img_size": int(img_size),
        "images_per_s": float(batch_size / (stats["p50"] / 1_000.0)),
        "torch": torch.__version__,
        "includes_preprocessing": False,
    })
    return stats


def tta_latency(model, k_views: int, **kw) -> dict:
    """Measure K actual forwards per call and compare with the single-view median."""
    if k_views < 1:
        raise ValueError("k_views must be positive")

    class TTAWrapper(nn.Module):
        def __init__(self, inner, count):
            super().__init__()
            self.inner = inner
            self.count = count

        def forward(self, x):
            outputs = [self.inner(x) for _ in range(self.count)]
            return torch.stack(outputs).mean(0)

    single = latency_report(model, **kw)
    measured = latency_report(TTAWrapper(model, k_views), **kw)
    expected = k_views * single["p50"]
    measured.update({
        "k_views": int(k_views),
        "single_view_p50": single["p50"],
        "linear_expected_p50": expected,
        "relative_to_single": measured["p50"] / single["p50"],
        "measured_over_linear": measured["p50"] / expected,
    })
    return measured
