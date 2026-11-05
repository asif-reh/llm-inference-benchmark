# LLM Inference Benchmark Lab

Public source for Shaik Asif Ur Rehman's client-observed LLM streaming benchmark.

- Dashboard: https://huggingface.co/spaces/Ki-Asif/inference-benchmark-lab
- Results: https://huggingface.co/datasets/Ki-Asif/inference-benchmark-results
- Real run: `results/local-cpu-smollm2-135m/` — SmolLM2-135M on local CPU, 24 measured requests, 0 errors.

Client-side toolkit for measuring LLM serving configurations and publishing
the result without inventing the parts you did not measure.

You point it at a backend, a prompt file, and a sweep of concurrency. It
records time to first token (TTFT), inter-token latency, end-to-end latency,
token and request throughput, errors, and timeouts. Warmup is stored and then
left out of the statistics. Accelerator memory is reported only when this
process, or the reference server process, can actually see the device.
Otherwise the field is null and the report says why.

The checked-in charts are a **synthetic example**, produced by a mock backend
so the pipeline and the dashboard run with no GPU and no API token. They are
not a measurement of a model server. A real figure comes from a config you
run, with the warnings and the sample size still attached.

Built by [Shaik Asif Ur Rehman](https://github.com/asif-reh) (Hugging Face:
[Ki-Asif](https://huggingface.co/Ki-Asif)), MSc Artificial Intelligence,
Dublin City University. The audience is AI infrastructure work: serving,
latency, and the difference between a client observation and a hardware claim.

## Quickstart

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make demo
make serve-dashboard
```

`make demo` runs `python -m benchlab run --config configs/example.yaml`. That
config is the offline mock. It writes:

- `results/example-synthetic/summary.json`
- `results/example-synthetic/summary.csv`
- `results/example-synthetic/requests.jsonl`
- copies of the summary for the static page and the Gradio app

The dashboard is then at <http://127.0.0.1:8741>. The yellow banner stays up
on purpose.

Check the CLI without sending traffic:

```bash
python -m benchlab --help
python -m benchlab run --config configs/example-hf.yaml --dry-run
python -m benchlab validate --config configs/example.yaml
```

`--dry-run` loads the config and the prompt file and exits. It does not
construct a client, so it does not need `HF_TOKEN`.

## Run against a real backend

Export a token only for the Inference API. The token is read from the
environment named in the config (`HF_TOKEN` by default). It is never written
into the result files.

```bash
export HF_TOKEN=hf_...
python -m benchlab run --config configs/example-hf.yaml
```

`configs/example-hf.yaml` is a template. Change `model_id` to a model your
token can call. `provider: hf-inference` keeps the call on the Hugging Face
Inference provider. Expect network time, cold starts, and shared capacity in
the numbers. The summary lists those limits. If the provider rejects
`temperature: 0`, set a small non-zero temperature and keep that fact in the
write-up. Sampling then becomes part of the latency spread, and the report
warns about it.

| Config | What it talks to |
| --- | --- |
| `configs/example.yaml` | Mock backend, virtual clock. No network. |
| `configs/example-hf.yaml` | `huggingface_hub.InferenceClient` streaming. |
| `configs/example-transformers.yaml` | `scripts/serve_transformers.py` on port 18080. |
| `configs/example-vllm.yaml` | vLLM `POST /v1/completions` stream. You start the server. |
| `configs/example-tgi.yaml` | TGI `POST /generate_stream`. You start the server. |

vLLM and TGI are thin clients behind the same adapter as the other backends.
They send one prompt per request. Batching stays a server setting
(`--max-num-seqs`, TGI max batch tokens). The config rejects a client batch
size other than 1 for those backends, so a result cannot imply a batch shape
the client never sent.

### Local transformers server

The reference server is for a real local run, not for CI. It serializes
`model.generate` with a lock. Concurrency includes queueing on that lock.
Say so if you publish the numbers. `sshleifer/tiny-gpt2` is a smoke test, not
a quality comparison.

```bash
pip install -e ".[local]"
python scripts/serve_transformers.py --model sshleifer/tiny-gpt2 --revision main --port 18080
python -m benchlab run --config configs/example-transformers.yaml
```

`GET /health` and `GET /vram` are on the same port. VRAM is omitted when CUDA
is not available.

Docker, still with no GPU on the default path:

```bash
docker compose up --build bench
```

The `transformers` service is behind the Compose profile `local` because the
image installs PyTorch. See comments in `docker-compose.yml`. Do not treat a
tiny-gpt2 CPU container as a serving benchmark.

## Architecture

```text
configs/*.yaml
    │
    ▼
benchlab.cli ──► config validation ──► backend adapter
                                         │
            ┌──────────────┬─────────────┼──────────────┬──────────────┐
            ▼              ▼             ▼              ▼              ▼
          mock      HF Inference   transformers      vLLM            TGI
       (virtual          API         HTTP client    HTTP client   HTTP client
        clock)
            │
            ▼
     closed-loop runner (warmup, then measure)
            │
            ▼
     summary.json + summary.csv + requests.jsonl
            │
            ▼
     space/  static dashboard
```

`Backend.generate` is the adapter. A new server is a new class that appends
token pieces to a `TokenSink` as the client observes them and returns a
`GenerationResult`. The runner owns concurrency, the timeout, warmup
exclusion, and the aggregate math, so backends cannot each invent a
percentile.

The load generator is closed-loop: at most `concurrency` requests are in
flight, and the next one starts when a slot frees. Latency starts when the
request is issued, not while it waits for a client slot. Server-side queueing
still shows up inside TTFT.

## Metrics

Full definitions, the percentile formula, and the publication checklist are
in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). The short form:

| Metric | What it is |
| --- | --- |
| TTFT | Issue → first streamed piece. |
| Inter-token latency | Gaps between later pieces, pooled. Mean, p50, p95. |
| End-to-end | Issue → stream complete. p50, p95, p99. |
| requests/s | Successful measured requests / measurement window. |
| output tokens/s | Streamed pieces from those requests / the same window. |
| Mean decode tokens/s | Per-request `(tokens - 1) / decode time`, then the mean. Use this for a single stream. |
| Error rate | Failed measured requests, including timeouts, over measured requests. |
| VRAM | Only if a probe ran on the process that holds the weights. |

Percentiles are Hyndman–Fan type 7, the same definition as NumPy's default.
p95 and p99 on a small N are labelled as unstable in `warnings`. Warmup is
not in any of those figures. Output tokens are streamed pieces as the server
sent them, not a second tokenization done locally.

Two throughput numbers are both reported because they answer different
questions. Overlapping requests raise output tokens per second of wall time
above the decode rate of any one request. Quoting only the larger one, as if
it were a single user's token rate, is a misread.

## Dashboard

`space/` is a static page. It fetches `data/example-synthetic.json`, draws
TTFT, throughput, a latency histogram and CDF, and renders the config, the
error table, and the methodology from the file itself. A file picker loads
any other benchlab `summary.json`. There is no chart CDN; the page works as
a static Space.

The Gradio alternative is `gradio_space/app.py`. It only displays the JSON.
Prefer the static page for hosting.

Publishing steps, the Space frontmatter, and the exact `hf upload` commands
are in [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md).

```bash
python scripts/publish_hf.py
```

That prints commands for Space `Ki-Asif/inference-benchmark-lab` and dataset
`Ki-Asif/inference-benchmark-results`. It does not upload.

## Tests

```bash
make test
```

That is ruff, mypy, and pytest. The tests cover percentile math, warmup
exclusion, config rejection (including virtual clock on a real backend, and
client batches on backends that do not send them), a scripted streaming
backend on the virtual clock, and the mock CLI. CI runs the same on Python
3.11 and 3.12. No GPU and no token.

## Layout

```text
benchlab/                 package (CLI, metrics, backends, export)
configs/                  example sweep and one template per backend
data/prompts/example.jsonl
results/example-synthetic/   SYNTHETIC fixture, labelled in-tree
space/                    static Hugging Face Space
gradio_space/             optional Gradio Space
scripts/serve_transformers.py
scripts/publish_hf.py     prints upload commands, never calls the Hub
docs/METHODOLOGY.md
docs/HUGGINGFACE.md
```

## Limitations

- A client timestamp includes the network when the server is remote. That is
  the measurement, not something to subtract after the fact without saying so.
- The Inference API path cannot see server VRAM, and it cannot pin a revision
  with the current `InferenceClient`. The report records the revision you
  wrote down and warns that it was not sent.
- The reference transformers server is one locked `generate` call. It is not
  vLLM and it is not TGI.
- vLLM and TGI clients do not configure server batching. They also do not
  probe the server's GPU.
- Closed-loop concurrency is not an open-loop arrival process.
- The synthetic fixture has injected failures so the error table is visible.
  Those rows are not incidents.
- Small N cannot support a tail SLO. The warning is in the JSON. Leave it
  there when you copy a chart out.

## License

MIT. Copyright 2026 Shaik Asif Ur Rehman.
