from __future__ import annotations

from pathlib import Path

import pytest
from benchlab.config import ConfigError, load_config
from benchlab.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load() -> None:
    for name in (
        "example.yaml",
        "example-hf.yaml",
        "example-transformers.yaml",
        "example-vllm.yaml",
        "example-tgi.yaml",
    ):
        loaded = load_config(ROOT / "configs" / name)
        assert len(loaded.prompts) == 20


def test_prompt_file_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    line = '{"id": "a", "category": "short_qa", "text": "hello"}\n'
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_prompts(path)


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_concurrency_and_backend_rules(tmp_path: Path) -> None:
    prompts = ROOT / "data" / "prompts" / "example.jsonl"
    output = tmp_path / "out"
    with pytest.raises(ConfigError, match="between 1 and 1024"):
        load_config(
            _write(
                tmp_path,
                "zero.yaml",
                f"""
schema_version: 1
run_name: bad
clock: virtual
backend:
  type: mock
  model_id: synthetic/mock-7b
workload:
  prompts: {prompts}
  concurrency: [0]
  batch_sizes: [1]
  warmup: 0
  samples: 1
  max_tokens: 8
  temperature: 0
  seed: 1
  timeout_s: 5
output:
  directory: {output}
""",
            )
        )
    with pytest.raises(ConfigError, match="fuse client-side batches"):
        load_config(
            _write(
                tmp_path,
                "hf.yaml",
                f"""
schema_version: 1
run_name: bad
clock: wall
backend:
  type: huggingface
  model_id: org/model
workload:
  prompts: {prompts}
  concurrency: [1]
  batch_sizes: [1, 2]
  warmup: 0
  samples: 1
  max_tokens: 8
  temperature: 0
  seed: 1
  timeout_s: 5
output:
  directory: {output}
""",
            )
        )
    with pytest.raises(ConfigError, match="virtual"):
        load_config(
            _write(
                tmp_path,
                "virt.yaml",
                f"""
schema_version: 1
run_name: bad
clock: virtual
backend:
  type: huggingface
  model_id: org/model
workload:
  prompts: {prompts}
  concurrency: [1]
  batch_sizes: [1]
  warmup: 0
  samples: 1
  max_tokens: 8
  temperature: 0
  seed: 1
  timeout_s: 5
output:
  directory: {output}
""",
            )
        )


def test_paths_resolve_relative_to_the_config_file() -> None:
    loaded = load_config(ROOT / "configs" / "example.yaml")
    assert loaded.prompts_path == (ROOT / "data" / "prompts" / "example.jsonl").resolve()
    assert loaded.output_dir == (ROOT / "results" / "example-synthetic").resolve()
