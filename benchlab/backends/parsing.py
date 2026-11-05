"""Pure parsers for streaming payloads. No network."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import cast


class StreamDone:
    """Sentinel: the SSE stream finished with ``data: [DONE]``."""


DONE = StreamDone()


def payload_from_sse_line(line: str) -> object | None:
    """Return a JSON object, ``DONE``, or None when the line carries no payload."""
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data:
        return None
    if data == "[DONE]":
        return DONE
    return _json_object(data)


def iter_sse_objects(lines: Iterator[str]) -> Iterator[dict[str, object]]:
    for line in lines:
        payload = payload_from_sse_line(line)
        if payload is None:
            continue
        if isinstance(payload, StreamDone):
            return
        if isinstance(payload, dict):
            yield payload


def token_from_text_generation(chunk: object) -> str | None:
    """Token text from an InferenceClient text-generation stream chunk.

    Strings are returned as-is. Object chunks use ``token.text`` and skip
    special tokens. Empty strings are skipped so they do not become a fake TTFT.
    """
    if isinstance(chunk, str):
        return chunk if chunk else None
    token = _child(chunk, "token")
    if token is None:
        return None
    if _child(token, "special") is True:
        return None
    text = _child(token, "text")
    if isinstance(text, str) and text:
        return text
    return None


def token_from_chat_chunk(chunk: object) -> str | None:
    """First non-empty ``delta.content`` on a chat completion stream chunk."""
    choices = _child(chunk, "choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        delta = _child(choice, "delta")
        content = _child(delta, "content") if delta is not None else None
        if isinstance(content, str) and content:
            return content
    return None


def texts_from_completion_chunk(payload: dict[str, object]) -> list[str]:
    """Token pieces from an OpenAI-style completions or chat chunk."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return []
    texts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        text = choice.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
    return texts


def text_from_tgi(payload: dict[str, object]) -> str | None:
    """Token text from a TGI ``/generate_stream`` event. Special tokens are skipped."""
    if "error" in payload and payload["error"]:
        raise ValueError(str(payload["error"]))
    token = payload.get("token")
    if not isinstance(token, dict):
        return None
    if token.get("special") is True:
        return None
    text = token.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def _child(value: object, name: str) -> object | None:
    if isinstance(value, dict):
        return cast(object, value.get(name))
    return cast(object, getattr(value, name, None))


def _json_object(data: str) -> dict[str, object]:
    parsed = cast(object, json.loads(data))
    if not isinstance(parsed, dict):
        raise ValueError("SSE payload is not a JSON object")
    output: dict[str, object] = {}
    for key, value in parsed.items():
        output[str(key)] = cast(object, value)
    return output
