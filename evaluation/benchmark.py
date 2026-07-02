from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn


OPERATION_COUNT_LIMITATION = (
    "Counts Conv2d, Linear, and MultiheadAttention only; excludes normalization, "
    "activation, indexing, pooling, and elementwise operations."
)


@dataclass(frozen=True)
class OperationCounts:
    macs: int
    flops: int
    limitation: str = OPERATION_COUNT_LIMITATION


@dataclass(frozen=True)
class LatencyMetrics:
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float
    peak_cuda_memory_mb: float


Shape = tuple[int, ...]


def estimate_module_macs(
    module: nn.Module,
    input_shape: Shape | tuple[Shape, Shape, Shape],
    output_shape: Shape,
) -> int:
    if isinstance(module, nn.Conv2d):
        batch, output_channels, output_height, output_width = output_shape
        kernel_height, kernel_width = module.kernel_size
        inputs_per_filter = (module.in_channels // module.groups) * kernel_height * kernel_width
        return int(
            batch * output_channels * output_height * output_width * inputs_per_filter
        )
    if isinstance(module, nn.Linear):
        return int(np.prod(output_shape) * module.in_features)
    if isinstance(module, nn.MultiheadAttention):
        query_shape, key_shape, _ = input_shape
        if module.batch_first:
            batch, query_length, embedding_dim = query_shape
            _, key_length, _ = key_shape
        else:
            query_length, batch, embedding_dim = query_shape
            key_length, _, _ = key_shape
        query_projection = batch * query_length * embedding_dim * embedding_dim
        key_value_projection = 2 * batch * key_length * embedding_dim * embedding_dim
        attention_products = 2 * batch * query_length * key_length * embedding_dim
        output_projection = batch * query_length * embedding_dim * embedding_dim
        return int(
            query_projection
            + key_value_projection
            + attention_products
            + output_projection
        )
    raise TypeError(f"Unsupported operation-count module: {type(module).__name__}")


def count_model_operations(model: nn.Module, sample_input: torch.Tensor) -> OperationCounts:
    total_macs = 0
    handles: list[torch.utils.hooks.RemovableHandle] = []
    mha_children = {
        id(child)
        for module in model.modules()
        if isinstance(module, nn.MultiheadAttention)
        for child in module.modules()
        if child is not module
    }

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: object) -> None:
        nonlocal total_macs
        if isinstance(module, nn.MultiheadAttention):
            query, key, value = inputs[:3]
            output_tensor = output[0] if isinstance(output, tuple) else output
            total_macs += estimate_module_macs(
                module,
                (tuple(query.shape), tuple(key.shape), tuple(value.shape)),
                tuple(output_tensor.shape),
            )
        else:
            output_tensor = output[0] if isinstance(output, tuple) else output
            total_macs += estimate_module_macs(
                module,
                tuple(inputs[0].shape),
                tuple(output_tensor.shape),
            )

    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, (nn.Conv2d, nn.Linear)) and id(module) not in mha_children:
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return OperationCounts(macs=total_macs, flops=2 * total_macs)


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def require_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for benchmarking but is not available")
    return device


def measure_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
    measure_iterations: int,
) -> LatencyMetrics:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measure_iterations < 1:
        raise ValueError("measure_iterations must be at least 1")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for benchmarking but is not available")
    model = model.to(device).eval()
    sample_input = sample_input.to(device)
    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(sample_input)

        timings: list[float] = []
        peak_memory = 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(measure_iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(sample_input)
                end.record()
                torch.cuda.synchronize(device)
                timings.append(float(start.elapsed_time(end)))
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            for _ in range(measure_iterations):
                start = time.perf_counter()
                model(sample_input)
                timings.append((time.perf_counter() - start) * 1000.0)

    mean_latency = float(np.mean(timings))
    return LatencyMetrics(
        mean_latency_ms=mean_latency,
        median_latency_ms=float(np.median(timings)),
        p95_latency_ms=float(np.percentile(timings, 95)),
        fps=1000.0 / mean_latency,
        peak_cuda_memory_mb=float(peak_memory),
    )


def write_runtime_metrics(
    path: str | Path,
    total_parameters: int,
    trainable_parameters: int,
    operations: OperationCounts,
    timing: LatencyMetrics,
    device: torch.device,
    warmup_iterations: int,
    measure_iterations: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "estimated_macs": operations.macs,
        "estimated_flops": operations.flops,
        "mean_latency_ms": timing.mean_latency_ms,
        "median_latency_ms": timing.median_latency_ms,
        "p95_latency_ms": timing.p95_latency_ms,
        "fps": timing.fps,
        "peak_cuda_memory_mb": timing.peak_cuda_memory_mb,
        "device": str(device),
        "warmup_iterations": warmup_iterations,
        "measure_iterations": measure_iterations,
        "operation_count_limitation": operations.limitation,
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
