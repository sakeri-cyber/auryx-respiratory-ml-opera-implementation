"""On-device deployment profiling: latency, memory, quantisation and ONNX export.

This is the part the original paper does not report. RespireNet (and OPERA after it)
optimise for benchmark score; neither publishes what the resulting model costs to
run. For anything that has to execute continuously on a wearable, that cost is the
binding constraint — a model that wins the benchmark by 2 points and does not fit in
the power budget has not won anything.

Measurements here are deliberately conservative:
  * batch size 1, since a wearable classifies one cycle at a time;
  * CPU by default, since that is the realistic target;
  * warm-up iterations discarded, and results reported as median + IQR rather than
    mean, because latency distributions are right-skewed and a mean flatters them.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LatencyResult:
    label: str
    median_ms: float
    p90_ms: float
    iqr_ms: float
    min_ms: float
    n_iterations: int
    batch_size: int
    device: str

    def __str__(self) -> str:
        return (
            f"{self.label:<28} median={self.median_ms:7.2f}ms  p90={self.p90_ms:7.2f}ms  "
            f"IQR={self.iqr_ms:6.2f}ms  (n={self.n_iterations}, bs={self.batch_size}, {self.device})"
        )


@dataclass(slots=True)
class SizeResult:
    label: str
    n_parameters: int
    disk_bytes: int

    @property
    def disk_mb(self) -> float:
        return self.disk_bytes / (1024 * 1024)

    def __str__(self) -> str:
        return f"{self.label:<28} params={self.n_parameters:>11,}  on disk={self.disk_mb:7.2f} MB"


def measure_latency(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    label: str = "model",
    n_warmup: int = 20,
    n_iterations: int = 200,
    device: str = "cpu",
) -> LatencyResult:
    """Time forward passes, discarding warm-up.

    Warm-up matters more than it looks: the first few passes pay for lazy kernel
    selection and allocator growth, and including them inflates the median by a
    factor that varies by machine — which is exactly how latency claims become
    irreproducible.
    """
    model = model.to(device).eval()
    example_input = example_input.to(device)

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(example_input)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            model(example_input)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    quartiles = statistics.quantiles(timings, n=4) if len(timings) >= 4 else [timings[0]] * 3
    return LatencyResult(
        label=label,
        median_ms=statistics.median(timings),
        p90_ms=timings[int(0.9 * (len(timings) - 1))],
        iqr_ms=quartiles[2] - quartiles[0],
        min_ms=timings[0],
        n_iterations=n_iterations,
        batch_size=example_input.shape[0],
        device=device,
    )


def measure_size(model: nn.Module, label: str = "model", tmp_dir: Path | None = None) -> SizeResult:
    """Parameter count and serialised size on disk."""
    tmp_dir = tmp_dir or Path(".profiling_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{label.replace('/', '_')}.pt"
    torch.save(model.state_dict(), path)
    size = path.stat().st_size
    path.unlink(missing_ok=True)
    return SizeResult(
        label=label,
        n_parameters=sum(p.numel() for p in model.parameters()),
        disk_bytes=size,
    )


def quantize_dynamic(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantisation of Linear layers.

    Note the honest limitation: dynamic quantisation only touches Linear (and RNN)
    layers. RespireNet is almost entirely Conv2d, so the win here is small — most of
    the model is untouched. Reporting that plainly is more useful than quietly
    picking the technique that produces the best-looking number.

    For a real deployment the next step is static quantisation with calibration data,
    or QAT, both of which do cover Conv2d. See `docs/` for the comparison.
    """
    return torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(), {nn.Linear}, dtype=torch.qint8
    )


def export_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    out_path: Path,
    *,
    opset: int = 17,
) -> Path:
    """Export to ONNX for runtimes that do not ship PyTorch."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        example_input.cpu(),
        str(out_path),
        input_names=["spectrogram"],
        output_names=["logits"],
        dynamic_axes={"spectrogram": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    logger.info("Exported ONNX to %s (%.2f MB)", out_path, out_path.stat().st_size / 1024**2)
    return out_path


def profile_report(
    model: nn.Module,
    example_input: torch.Tensor,
    *,
    out_path: Path | None = None,
    include_quantized: bool = True,
) -> dict:
    """Run the full profile and return a JSON-serialisable report.

    Deliberately records the host description alongside the numbers. Latency figures
    without the machine they were measured on are not a result, they are a rumour.
    """
    report: dict = {
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "input_shape": list(example_input.shape),
        "latency": [],
        "size": [],
    }

    fp32_latency = measure_latency(model, example_input, label="fp32")
    fp32_size = measure_size(model, label="fp32")
    report["latency"].append(asdict(fp32_latency))
    report["size"].append(asdict(fp32_size))
    logger.info("%s", fp32_latency)
    logger.info("%s", fp32_size)

    if include_quantized:
        try:
            quantized = quantize_dynamic(model)
            q_latency = measure_latency(quantized, example_input, label="int8-dynamic")
            q_size = measure_size(quantized, label="int8-dynamic")
            report["latency"].append(asdict(q_latency))
            report["size"].append(asdict(q_size))
            report["quantisation_note"] = (
                "Dynamic quantisation covers Linear layers only. RespireNet is "
                "predominantly Conv2d, so the observed gain is small by construction."
            )
            logger.info("%s", q_latency)
            logger.info("%s", q_size)
        except Exception as exc:  # noqa: BLE001 - report rather than abort the profile
            logger.warning("Quantisation failed: %s", exc)
            report["quantisation_error"] = str(exc)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        logger.info("Wrote profile report to %s", out_path)

    return report
