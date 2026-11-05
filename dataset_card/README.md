---
license: mit
tags:
  - benchmark
  - llm-inference
pretty_name: LLM Inference Benchmark Results
---

# LLM inference benchmark results

Artifacts produced by [benchlab](https://github.com/asif-reh): `summary.json`,
`summary.csv`, and `requests.jsonl` for one configuration sweep.

## SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK

The folder `example-synthetic/` is the output of the mock backend. Token times
come from a discrete-event latency model so the dashboard can be reviewed
without a GPU or an API token. Those files are not measurements of a model,
a GPU, or the Hugging Face Inference API.

Replace that folder with a real run before you cite this dataset. A real run
sets `meta.synthetic` to false and `notice` to null. Read `meta`, `hardware`,
`warnings`, and `limitations` in `summary.json` before repeating a number.
Warmup requests are in `requests.jsonl` and are excluded from the summary
statistics. VRAM fields are null when the client could not see the serving
device.
