#!/usr/bin/env python3
"""Reference streaming server for the ``transformers`` backend.

This is not a production server. ``model.generate`` is held under one lock, so
concurrent benchmark clients queue in this process. Use it to check the client
and, if you have a GPU, to record an honest local number. ``tiny-gpt2`` is a
connectivity smoke test, not a model you should publish as a quality result.

    pip install -e ".[local]"
    python scripts/serve_transformers.py --model sshleifer/tiny-gpt2 --revision main

Then, in another shell:

    python -m benchlab run --config configs/example-transformers.yaml

``GET /health`` and ``GET /vram`` report process state. VRAM stays
``measurable: false`` when CUDA is not available. ``POST /generate`` streams
server-sent events::

    data: {"text": "...", "index": 0}

    data: {"done": true, "output_tokens": 12}
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local transformers streaming server for benchlab."
    )
    parser.add_argument("--model", required=True, help="model id or local path")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args(argv)
    try:
        bundle = _load(args.model, args.revision, args.device)
    except ImportError:
        print(
            "transformers and torch are not installed. Install the local extra:\n"
            '  pip install -e ".[local]"',
        )
        return 1

    handler = _make_handler(bundle, args.model, args.revision)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"benchlab transformers reference server on http://{args.host}:{args.port}")
    print("generation is serialized by a lock; this is not a production scheduler")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


def _load(model_id: str, revision: str | None, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return {"model": model, "tokenizer": tokenizer, "device": device, "lock": threading.Lock()}


def _make_handler(
    bundle: dict[str, Any], model_id: str, revision: str | None
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: object) -> None:
            print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % log_args}")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/health":
                self._json({"status": "ok", "model_id": model_id, "revision": revision})
                return
            if path == "/vram":
                from benchlab.vram import probe_local_vram

                self._json(probe_local_vram().as_dict())
                return
            self._json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path != "/generate":
                self._json({"error": "not found"}, status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "invalid JSON"}, status=400)
                return
            prompts = payload.get("prompts")
            if prompts is None and isinstance(payload.get("prompt"), str):
                prompts = [payload["prompt"]]
            if (
                not isinstance(prompts, list)
                or not prompts
                or not all(isinstance(p, str) for p in prompts)
            ):
                self._json({"error": "prompts must be a non-empty list of strings"}, status=400)
                return
            max_new = payload.get("max_new_tokens", 32)
            if not isinstance(max_new, int) or max_new < 1 or max_new > 4096:
                self._json(
                    {"error": "max_new_tokens must be an integer from 1 to 4096"}, status=400
                )
                return
            temperature = payload.get("temperature", 0)
            if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
                self._json({"error": "temperature must be a number"}, status=400)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Benchlab-Server", "serial")
            self.end_headers()
            produced = 0
            try:
                for event in _generate(bundle, prompts, max_new, float(temperature)):
                    produced += 1
                    self.wfile.write(_sse(event))
                    self.wfile.flush()
            except Exception as exc:
                self.wfile.write(_sse({"error": f"{type(exc).__name__}: {exc}"}))
                self.wfile.flush()
            self.wfile.write(_sse({"done": True, "output_tokens": produced}))
            self.wfile.flush()

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _generate(
    bundle: dict[str, Any], prompts: list[str], max_new_tokens: int, temperature: float
) -> Any:
    import torch
    from transformers import TextIteratorStreamer

    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    device = bundle["device"]
    index = 0
    for prompt in prompts:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Instruct models need their chat template; raw prompts often emit EOS
        # immediately, which the runner correctly reports as empty streams.
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_kwargs: dict[str, Any] = dict(inputs)
        generate_kwargs["streamer"] = streamer
        generate_kwargs["max_new_tokens"] = max_new_tokens
        if temperature <= 0:
            generate_kwargs["do_sample"] = False
        else:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature

        def _run(kwargs: dict[str, Any] = generate_kwargs) -> None:
            with torch.inference_mode():
                model.generate(**kwargs)

        with bundle["lock"]:
            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            for text in streamer:
                if text:
                    yield {"text": text, "index": index}
                    index += 1
            worker.join()


def _sse(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


if __name__ == "__main__":
    raise SystemExit(main())
