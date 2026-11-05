"""Accelerator memory probes.

A reading is omitted (measurable is false) when this process cannot see the
device that actually served the model. Remote APIs do not become GPU
benchmarks because the client machine has a GPU.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class VramReading:
    measurable: bool
    peak_allocated_mb: float | None
    instantaneous_used_mb: float | None
    source: str | None
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "measurable": self.measurable,
            "peak_allocated_mb": self.peak_allocated_mb,
            "instantaneous_used_mb": self.instantaneous_used_mb,
            "source": self.source,
            "note": self.note,
        }

    @staticmethod
    def not_measurable(note: str) -> VramReading:
        return VramReading(
            measurable=False,
            peak_allocated_mb=None,
            instantaneous_used_mb=None,
            source=None,
            note=note,
        )


def probe_local_vram() -> VramReading:
    """Probe this process. ``peak_allocated_mb`` is torch CUDA peak allocation.

    ``instantaneous_used_mb`` is the max ``memory.used`` nvidia-smi reports
    across visible GPUs at call time. It is not a peak over the run unless the
    caller samples at the peak. Both fields stay null when the tool is absent.
    """
    instantaneous = _nvidia_smi_used_mb()
    peak = _torch_peak_allocated_mb()
    sources: list[str] = []
    if instantaneous is not None:
        sources.append("nvidia-smi")
    if peak is not None:
        sources.append("torch.cuda.max_memory_allocated")
    if not sources:
        return VramReading.not_measurable(
            "VRAM is not measurable here: nvidia-smi is unavailable and torch CUDA is not active."
        )
    note = (
        "Local process probe. nvidia-smi is an instantaneous reading, not a "
        "peak over the run. torch peak is the allocator high-water mark since "
        "the last reset in this process."
    )
    return VramReading(
        measurable=True,
        peak_allocated_mb=peak,
        instantaneous_used_mb=instantaneous,
        source="+".join(sources),
        note=note,
    )


def _nvidia_smi_used_mb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    values: list[float] = []
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text or text.lower() == "n/a":
            continue
        try:
            values.append(float(text.split()[0]))
        except ValueError:
            return None
    if not values:
        return None
    return round(max(values), 2)


def _torch_peak_allocated_mb() -> float | None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return None
    peak_bytes = int(cuda.max_memory_allocated())
    return round(peak_bytes / (1024 * 1024), 2)
