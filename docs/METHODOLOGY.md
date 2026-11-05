# Methodology

benchlab measures a serving configuration from the client that sends the
traffic. It records what that client observed. It does not fill in a GPU
number when the device is on another machine, and it does not treat the mock
backend as a model.

The short definitions in a run's `summary.json` under `definitions` are the
same sentences the code copies into every report. This document is the
operating version: how a run is constructed, what is excluded, and what a
figure is not allowed to mean.

## What a run is

A config names a backend, a model id, an optional revision, a prompt file, and
a sweep of concurrency levels and client batch sizes. For each pair of
`(concurrency, batch_size)` the runner does two phases:

1. **Warmup.** `workload.warmup` requests at that concurrency. Stored in
   `requests.jsonl` with `phase: warmup`. Excluded from every summary
   statistic, including error rate.
2. **Measure.** `workload.samples` requests. These are the only rows that
   enter TTFT, inter-token latency, end-to-end latency, throughput, and
   `error_rate`.

Prompts are assigned in issue order from a cycle of the prompt file. When
`workload.seed` is set, the file is shuffled once with that seed before the
cycle starts. The same seed is forwarded to backends whose client accepts a
`seed` argument (Hugging Face text generation and chat completion, vLLM, TGI).
The mock backend also uses it for its latency model. A null seed keeps file
order and is not sent as a generation seed.

`prompts.sha256` in the summary is the SHA-256 of the prompt file bytes.
Publish the prompt file, or at least that hash, with the result. The summary
stores prompt ids, not prompt text.

## Clocks

`clock: wall` uses `time.perf_counter` on the client. Offsets in the JSONL
(`t_start_us`, `token_offset_ms`) are microseconds from the start of the run,
not Unix timestamps. `meta.created_at` is the wall time the report was built.

`clock: virtual` is allowed only for the mock backend. It is a discrete-event
clock in integer microseconds. Concurrent requests overlap in that simulated
time. A virtual run is deterministic given the config and the prompt file. It
is how `results/example-synthetic/` is produced. `created_at` on that fixture
is frozen; `meta.created_at_is_frozen` says so. Do not read the frozen
timestamp as the time a server was measured.

## Load pattern

The generator is **closed-loop**. A semaphore of size `concurrency` keeps at
most that many requests in flight. The next request starts when a slot frees.
Latency clocks start after the slot is acquired, at the moment the request is
issued. Time spent waiting for a client slot is not part of TTFT or
end-to-end latency.

Queueing inside the server still appears in TTFT. That is intentional.

This is not an open-loop Poisson arrival process. Do not describe the result
as one.

Client batch size is separate from concurrency:

- **mock** fuses a batch into one simulated stream. The first prompt pays a
  small extra TTFT per extra prompt; later prompts are appended. This exists
  so the sweep and the metrics can be tested without a server. It is still
  synthetic.
- **transformers** (the reference server in `scripts/serve_transformers.py`)
  accepts a list of prompts as one HTTP request and generates them in order
  under a process-wide lock.
- **huggingface, vLLM, and TGI** reject `batch_sizes` other than `[1]`. Those
  servers batch internally. Set that on the server (`--max-num-seqs`, TGI max
  batch tokens) and put load on `concurrency`. The client does not pretend to
  have configured it.

## Metrics

All latency statistics use successful measured requests only. Failed requests,
including timeouts, stay in the measurement window and in `error_rate`.

| Name | Definition |
| --- | --- |
| TTFT | Time from request issue to the first streamed token, observed by this client. The clock starts after a concurrency slot is acquired. |
| Inter-token latency | Gaps between consecutive streamed tokens, pooled across successful measured requests. The first token is TTFT and is not an inter-token gap. A request with one token contributes no inter-token samples. Reported as mean, p50, and p95 of that pooled sample. `itl_ms.n` is the number of gaps, not the number of requests. |
| End-to-end latency | Time from request issue until the client observes stream completion, including the last token. Reported as mean, p50, p95, p99, min, and max. |
| Measurement window | On the run clock, from the earliest measured request issue to the latest measured request finish at that level. Warmup is outside the window. Failures are inside it when they occupied a slot. |
| requests/s | Successful measured requests divided by the measurement window in seconds. |
| output tokens/s | Streamed pieces from successful measured requests, divided by the same window. |
| Mean decode tokens/s | Mean, over successful requests with at least two tokens, of `(output_tokens - 1) / (time from first token to last token)`. This is a per-stream decode rate. The windowed output tokens/s is a system rate and is usually higher once requests overlap. Use the decode rate to talk about a single user's stream. |
| Output tokens | Count of non-empty streamed pieces. This follows the provider's stream tokenization. benchlab does not retokenize with a local tokenizer, because that count can disagree with the server and would hide what the client actually received. |
| Error rate | Failed measured requests, including timeouts, divided by measured requests issued. Timeouts are also broken out as `n_timeouts`. Warmup failures are `n_warmup_errors` and `warmup_errors`. They are not in `error_rate`. |

Percentiles are **Hyndman–Fan type 7**: linear interpolation between the two
nearest ranks, `rank = (q / 100) * (n - 1)`. This is NumPy's default
percentile. The summary records `meta.percentile_method: hyndman_fan_type7`.

With a small N the tails describe the sample, not a stable service level.
The report adds a warning when successful N is below 20 (p95 is coarse) and
when it is below 100 (p99 is not stable). p99 is still present so you can see
the sample. Do not drop it into an SLO.

The end-to-end histogram uses Sturges' rule, equal width from the sample min
to the sample max. Counts sum to successful N. A constant sample is one bin.
The CDF is the empirical distribution, `fraction = rank / n` with 1-based
ranks, downsampled to at most 200 points. The raw offsets remain in
`requests.jsonl`.

## Errors and timeouts

Each request has a client timeout (`workload.timeout_s`). On expiry the
in-flight call is cancelled. Latency fields on that JSONL row are null
(`stats_included: false`). Any pieces observed before cancellation are kept
as `token_offset_ms` and `output_tokens` so the row can be audited, and they
are not in the latency pool.

`empty_stream` means the backend returned success with no token pieces.
`backend` is any other client or server failure. The message is the exception
text, truncated for HTTP bodies.

A run that finishes with errors still exits 0. The errors are the result.
A config that cannot be loaded exits 2.

## VRAM

`hardware.measurable` is false unless a probe actually ran against the process
that holds the weights.

- Mock: not applicable.
- Hugging Face Inference API, vLLM client, TGI client: the client does not
  probe its own machine and call it the server. The note says so. Record
  `nvidia-smi` or the server's own metrics on the serving host if you want to
  publish memory.
- Reference transformers server: `GET /vram` runs the probe in that process.
  `peak_allocated_mb` is `torch.cuda.max_memory_allocated` when CUDA is
  active. `instantaneous_used_mb` is the maximum `memory.used` nvidia-smi
  reports across visible GPUs at call time. nvidia-smi is not a peak over the
  run. On CPU both stay null and `measurable` is false.

## Backends, in one paragraph each

**Hugging Face Inference API.** `huggingface_hub.InferenceClient`, streaming.
`mode: chat` uses `chat_completion` and counts `delta.content` only.
Reasoning-channel text is ignored when the provider leaves it out of
`content`. `mode: text` uses `text_generation` with `details=True` and skips
special tokens. `provider: hf-inference` keeps the call on the HF Inference
provider; `auto` lets the client route. The installed client does not pin a
revision on that API. The revision field is recorded, and the summary warns
that it was not sent. Timestamps are taken when the asyncio thread receives a
piece, so TTFT includes a small scheduling delay plus the network.

**transformers.** HTTP client of `scripts/serve_transformers.py`. The server
loads `AutoModelForCausalLM` and streams with `TextIteratorStreamer`. One lock
around `generate`. Concurrency measures that queue plus compute. Say that
when you publish.

**vLLM.** `POST {base_url}/v1/completions` with `stream: true`. One prompt per
request. An API key is read from `VLLM_API_KEY` only when the variable is set.

**TGI.** `POST {base_url}/generate_stream`. One input per request. Temperature
0 is sent as `do_sample: false` rather than `temperature: 0`, which some TGI
builds reject.

**mock.** A seeded function of prompt id, category, concurrency, and ordinal.
Higher concurrency increases TTFT and inter-token gaps so a sweep is not a
flat line. That slope is part of the model, not a finding. Injected errors in
the example config exist so the dashboard's error table is not an empty
state. The summary notice is `SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK`.

## What you should publish with a real number

- The config file, or the `config` object from `summary.json`.
- Prompt file hash, warmup, N per level, seed, temperature, max tokens, timeout.
- Backend version and, for a local server, the image digest or the exact
  command line, including the model revision actually loaded.
- Hardware only from the machine that served, including a note when VRAM was
  an instantaneous nvidia-smi reading.
- The warnings list. If p99 was flagged unstable, do not quote p99 alone.

## What this will not do

It will not invent a throughput figure for a GPU it did not run on. It will
not average TTFT across concurrency levels into one number. It will not drop
failed requests and then report the error rate as zero. It will not call the
checked-in synthetic fixture a benchmark result.
