"""Gradio viewer for a benchlab summary JSON.

The static Space under ``space/`` is the one to publish for free CPU hosting.
Use this file only if you create a separate Gradio Space. It does not run a
benchmark and it does not call a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

DEFAULT = Path(__file__).resolve().parent / "data" / "example-synthetic.json"
NOTICE = "SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK"


def present(path: str) -> tuple[str, list[list[object]], list[list[object]], str]:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return f"File not found: {file_path}", [], [], ""
    try:
        summary = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"Invalid JSON: {exc}", [], [], ""
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        return "Not a benchlab summary (schema_version 1).", [], [], ""

    meta = summary.get("meta") if isinstance(summary.get("meta"), dict) else {}
    synthetic = bool(meta.get("synthetic") or summary.get("notice"))
    headline = NOTICE if synthetic else f"{meta.get('backend', '')} · {meta.get('model_id', '')}"
    if synthetic:
        headline += (
            "\n\nThese figures come from the mock backend's discrete-event clock. "
            "They are not a measurement of a model server."
        )

    rows: list[list[object]] = []
    for level in summary.get("levels") or []:
        if not isinstance(level, dict):
            continue
        ttft = level.get("ttft_ms") if isinstance(level.get("ttft_ms"), dict) else {}
        e2e = level.get("e2e_ms") if isinstance(level.get("e2e_ms"), dict) else {}
        rows.append(
            [
                level.get("concurrency"),
                level.get("n_ok"),
                level.get("n_measured"),
                level.get("error_rate"),
                ttft.get("p50"),
                ttft.get("p95"),
                e2e.get("p50"),
                e2e.get("p95"),
                e2e.get("p99"),
                level.get("output_tokens_per_sec"),
                level.get("requests_per_sec"),
            ]
        )

    errors: list[list[object]] = []
    for row in summary.get("errors") or []:
        if isinstance(row, dict):
            errors.append(
                [
                    row.get("request_id"),
                    row.get("concurrency"),
                    row.get("error_type"),
                    row.get("message"),
                ]
            )

    hardware = summary.get("hardware") if isinstance(summary.get("hardware"), dict) else {}
    lines = [
        f"Run `{meta.get('run_id')}` · created {meta.get('created_at')}",
        "Clock `{}` · seed `{}` · percentile `{}`".format(
            meta.get("clock"), meta.get("seed"), meta.get("percentile_method")
        ),
        f"VRAM measurable: {hardware.get('measurable')}. {hardware.get('note', '')}",
        "",
        "Warnings:",
    ]
    warnings = summary.get("warnings") if isinstance(summary.get("warnings"), list) else []
    lines.extend(f"- {item}" for item in warnings)
    lines.append("")
    lines.append("Limitations:")
    limitations = summary.get("limitations") if isinstance(summary.get("limitations"), list) else []
    lines.extend(f"- {item}" for item in limitations)
    return headline, rows, errors, "\n".join(str(line) for line in lines)


with gr.Blocks(title="LLM Inference Benchmark Lab") as demo:
    gr.Markdown(
        "# LLM Inference Benchmark Lab\n"
        "Read a `summary.json` from benchlab. This app does not generate numbers."
    )
    path = gr.Textbox(label="Summary JSON path", value=str(DEFAULT))
    load = gr.Button("Load")
    headline = gr.Markdown()
    levels = gr.Dataframe(
        headers=[
            "concurrency",
            "n_ok",
            "n_measured",
            "error_rate",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "e2e_p50_ms",
            "e2e_p95_ms",
            "e2e_p99_ms",
            "output_tokens_per_sec",
            "requests_per_sec",
        ],
        label="Levels (warmup excluded)",
    )
    errors = gr.Dataframe(
        headers=["request_id", "concurrency", "error_type", "message"],
        label="Measured errors",
    )
    notes = gr.Textbox(label="Methodology for this file", lines=16)
    load.click(present, inputs=path, outputs=[headline, levels, errors, notes])

if __name__ == "__main__":
    demo.launch()
