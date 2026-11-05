from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

from benchlab.definitions import SYNTHETIC_NOTICE

ROOT = Path(__file__).resolve().parents[1]


class _Ids(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key == "id" and value:
                self.ids.add(value)


def test_static_space_has_the_sections_and_the_synthetic_label() -> None:
    html = (ROOT / "space" / "index.html").read_text(encoding="utf-8")
    parser = _Ids()
    parser.feed(html)
    required = {
        "banner",
        "kpis",
        "config-table",
        "chart-ttft",
        "chart-throughput",
        "chart-hist",
        "chart-cdf",
        "latency-table",
        "error-table",
        "methodology",
        "file-input",
    }
    assert required <= parser.ids
    assert SYNTHETIC_NOTICE in html
    script = (ROOT / "space" / "app.js").read_text(encoding="utf-8")
    assert "e2e_histogram" in script
    assert "textContent" in script
    summary = json.loads((ROOT / "space" / "data" / "example-synthetic.json").read_text())
    assert summary["notice"] == SYNTHETIC_NOTICE
    assert summary["meta"]["synthetic"] is True
    assert summary["schema_version"] == 1
    readme = (ROOT / "results" / "example-synthetic" / "README.md").read_text(encoding="utf-8")
    assert SYNTHETIC_NOTICE in readme
    space_readme = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
    assert "sdk: static" in space_readme
    assert SYNTHETIC_NOTICE in space_readme
