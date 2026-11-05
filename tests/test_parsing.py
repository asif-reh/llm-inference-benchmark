from __future__ import annotations

from types import SimpleNamespace

import pytest
from benchlab.backends.parsing import (
    DONE,
    StreamDone,
    payload_from_sse_line,
    text_from_tgi,
    texts_from_completion_chunk,
    token_from_chat_chunk,
    token_from_text_generation,
)
from benchlab.records import ErrorType, GenerationResult, coerce_result
from benchlab.vram import probe_local_vram


def test_text_generation_chunk_skips_special_and_empty() -> None:
    assert token_from_text_generation("Hi") == "Hi"
    assert token_from_text_generation("") is None
    chunk = SimpleNamespace(token=SimpleNamespace(text="tok", special=False))
    assert token_from_text_generation(chunk) == "tok"
    special = SimpleNamespace(token=SimpleNamespace(text="<s>", special=True))
    assert token_from_text_generation(special) is None
    assert token_from_text_generation({"token": {"text": "x", "special": False}}) == "x"


def test_chat_chunk_uses_content_only() -> None:
    role_only = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
    assert token_from_chat_chunk(role_only) is None
    text = {"choices": [{"delta": {"content": "Hello"}}]}
    assert token_from_chat_chunk(text) == "Hello"


def test_sse_and_server_payloads() -> None:
    assert payload_from_sse_line(": keep-alive") is None
    assert payload_from_sse_line("data: [DONE]") is DONE
    assert isinstance(payload_from_sse_line("data: [DONE]"), StreamDone)
    payload = payload_from_sse_line('data: {"text": "ab"}')
    assert payload == {"text": "ab"}
    assert texts_from_completion_chunk(
        {"choices": [{"text": "a"}, {"delta": {"content": "b"}}]}
    ) == [
        "a",
        "b",
    ]
    assert text_from_tgi({"token": {"text": "z", "special": False}}) == "z"
    assert text_from_tgi({"token": {"text": "</s>", "special": True}}) is None
    with pytest.raises(ValueError, match="boom"):
        text_from_tgi({"error": "boom"})


def test_coerce_empty_stream() -> None:
    result = coerce_result(GenerationResult(ok=True, started_us=10, finished_us=20), 10)
    assert result.ok is False
    assert result.error_type == ErrorType.EMPTY_STREAM


def test_vram_probe_does_not_raise() -> None:
    reading = probe_local_vram()
    assert reading.measurable in {True, False}
    assert reading.note
