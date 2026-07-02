from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.benchmark import (  # noqa: E402
    OPERATION_COUNT_LIMITATION,
    count_model_operations,
    estimate_module_macs,
    measure_latency,
    parameter_counts,
    require_device,
    write_runtime_metrics,
)
from scripts.benchmark_wipose import parse_args  # noqa: E402


def test_linear_mac_count() -> None:
    layer = nn.Linear(4, 3, bias=True)
    assert estimate_module_macs(layer, (2, 5, 4), (2, 5, 3)) == 2 * 5 * 3 * 4


def test_conv2d_mac_count() -> None:
    layer = nn.Conv2d(3, 8, kernel_size=3)
    assert estimate_module_macs(layer, (1, 3, 10, 10), (1, 8, 8, 8)) == (
        1 * 8 * 8 * 8 * 3 * 3 * 3
    )


def test_multihead_attention_mac_count_includes_projection_and_attention() -> None:
    layer = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    expected = (
        1 * 3 * 8 * 8
        + 2 * 1 * 5 * 8 * 8
        + 2 * 1 * 3 * 5 * 8
        + 1 * 3 * 8 * 8
    )
    assert estimate_module_macs(
        layer,
        ((1, 3, 8), (1, 5, 8), (1, 5, 8)),
        (1, 3, 8),
    ) == expected


def test_model_operation_count_is_deterministic_and_avoids_mha_double_count() -> None:
    class TinyAttentionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(4, 8)
            self.attention = nn.MultiheadAttention(8, 2, batch_first=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            projected = self.projection(x)
            return self.attention(projected, projected, projected, need_weights=False)[0]

    model = TinyAttentionModel()
    sample = torch.randn(1, 3, 4)

    first = count_model_operations(model, sample)
    second = count_model_operations(model, sample)

    projection_macs = 1 * 3 * 8 * 4
    attention_macs = 4 * (1 * 3 * 8 * 8) + 2 * (1 * 3 * 3 * 8)
    assert first.macs == projection_macs + attention_macs
    assert first == second
    assert first.flops == first.macs * 2
    assert first.limitation == OPERATION_COUNT_LIMITATION


def test_cpu_latency_and_parameter_metrics(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
    sample = torch.randn(1, 4)

    timing = measure_latency(
        model,
        sample,
        device=torch.device("cpu"),
        warmup_iterations=2,
        measure_iterations=5,
    )
    total, trainable = parameter_counts(model)
    path = tmp_path / "runtime_metrics.csv"
    write_runtime_metrics(
        path,
        total_parameters=total,
        trainable_parameters=trainable,
        operations=count_model_operations(model, sample),
        timing=timing,
        device=torch.device("cpu"),
        warmup_iterations=2,
        measure_iterations=5,
    )

    assert timing.mean_latency_ms > 0
    assert timing.median_latency_ms > 0
    assert timing.p95_latency_ms > 0
    assert timing.fps == pytest.approx(1000.0 / timing.mean_latency_ms)
    assert timing.peak_cuda_memory_mb == 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert int(row["total_parameters"]) == total
    assert row["operation_count_limitation"] == OPERATION_COUNT_LIMITATION


def test_cuda_request_never_silently_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        require_device("cuda")
    assert require_device("cpu") == torch.device("cpu")


def test_benchmark_cli_requires_checkpoint_manifest_and_key() -> None:
    args = parse_args([
        "--dataset-root", "dataset",
        "--checkpoint", "model.pth",
        "--split-manifest", "split.npz",
        "--manifest-key", "env2_test",
        "--output-dir", "output",
        "--device", "cpu",
    ])

    assert args.checkpoint == Path("model.pth")
    assert args.manifest_key == "env2_test"
    assert args.warmup_iterations == 20
    assert args.measure_iterations == 100
