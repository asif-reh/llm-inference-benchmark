from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from benchlab.cli import main
from benchlab.definitions import SYNTHETIC_NOTICE

ROOT = Path(__file__).resolve().parents[1]


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    with pytest.raises(SystemExit) as caught_run:
        main(["run", "--help"])
    assert caught_run.value.code == 0


def test_dry_run_and_validate_do_not_need_a_token(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--config", str(ROOT / "configs" / "example-hf.yaml")]) == 0
    validate_out = capsys.readouterr().out
    assert "valid" in validate_out
    assert "huggingface" in validate_out
    assert main(["run", "--config", str(ROOT / "configs" / "example-hf.yaml"), "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert "dry-run: no requests sent" in dry
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in dry


def test_mock_cli_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        [
            "run",
            "--config",
            str(ROOT / "configs" / "example.yaml"),
            "--output",
            str(tmp_path / "run"),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert SYNTHETIC_NOTICE in text
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["synthetic"] is True
    assert summary["meta"]["clock"] == "virtual"
    assert summary["hardware"]["measurable"] is False
    assert len(summary["levels"]) == 4
    assert any(row["error_type"] == "timeout" for row in summary["errors"])
    assert any(row["error_type"] == "backend" for row in summary["errors"])
    # Warmup is retained in the raw log and excluded from the per-level N.
    lines = (tmp_path / "run" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["phase"] == "warmup" for line in lines)
    assert sum(level["n_measured"] for level in summary["levels"]) == 160
    assert "p99 is reported for completeness" in " ".join(summary["warnings"])


def test_reference_server_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "serve_transformers.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "generate" in completed.stdout.lower() or "model" in completed.stdout.lower()


def test_publish_script_does_not_upload() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "publish_hf.py"), "--execute"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    printed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "publish_hf.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert printed.returncode == 0
    assert "Ki-Asif/inference-benchmark-lab" in printed.stdout
    assert "Ki-Asif/inference-benchmark-results" in printed.stdout
    assert "--repo-type space" in printed.stdout
    assert "--repo-type dataset" in printed.stdout
