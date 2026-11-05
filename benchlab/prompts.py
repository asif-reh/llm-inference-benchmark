"""Prompt-set loader.

Files are JSONL. Each line is an object with ``id``, ``category``, and ``text``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    id: str
    category: str
    text: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_prompts(path: Path) -> list[Prompt]:
    if not path.is_file():
        raise ValueError(f"prompt file not found: {path}")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        prompt_id = payload.get("id")
        category = payload.get("category")
        text = payload.get("text")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"{path}:{line_number}: id must be a non-empty string")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{path}:{line_number}: category must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: text must be a non-empty string")
        if prompt_id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate prompt id {prompt_id}")
        seen.add(prompt_id)
        prompts.append(Prompt(id=prompt_id, category=category.strip(), text=text))
    if not prompts:
        raise ValueError(f"prompt file is empty: {path}")
    return prompts
