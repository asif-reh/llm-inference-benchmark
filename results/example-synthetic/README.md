# SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK

This directory is the output of `configs/example.yaml`.

The mock backend simulates token arrival with a discrete-event clock. Nothing
in `summary.json`, `summary.csv`, or `requests.jsonl` was measured on a GPU,
on the Hugging Face Inference API, or on a vLLM / TGI / transformers server.

`meta.synthetic` is true. `meta.created_at` is frozen so the fixture stays
byte-stable. The two failed requests are injected so the dashboard error
table has something to show.

Regenerate with:

```bash
make demo
```
