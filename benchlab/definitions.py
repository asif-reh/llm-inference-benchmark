"""Canonical wording for metrics. The summary JSON copies these strings."""

from __future__ import annotations

SYNTHETIC_NOTICE = "SYNTHETIC EXAMPLE — NOT A REAL BENCHMARK"

PERCENTILE_METHOD = "hyndman_fan_type7"

DEFINITIONS: dict[str, str] = {
    "ttft_ms": (
        "Time from request issue to the first streamed token, observed by this client. "
        "The clock starts after a concurrency slot is acquired."
    ),
    "itl_ms": (
        "Gaps between consecutive streamed tokens, pooled across successful measured "
        "requests. The first token is TTFT and is not an inter-token gap. Requests "
        "with one token contribute no inter-token samples."
    ),
    "e2e_ms": (
        "Time from request issue until the client observes stream completion, "
        "including the last token."
    ),
    "requests_per_sec": (
        "Successful measured requests divided by the measurement window in seconds. "
        "Failed requests are excluded from the numerator and included in the window "
        "when they occupied a concurrency slot."
    ),
    "output_tokens_per_sec": (
        "Streamed token pieces from successful measured requests, divided by the "
        "measurement window in seconds."
    ),
    "mean_decode_tokens_per_sec": (
        "Mean, over successful requests with at least two tokens, of "
        "(output_tokens - 1) / (time from the first token to the last token)."
    ),
    "measurement_window_s": (
        "On the run clock, earliest measured request issue to latest measured request "
        "finish at that concurrency level. Warmup is outside the window. The load "
        "pattern is closed-loop: at most `concurrency` requests are in flight, and a "
        "new request starts when a slot frees. Client-side queue delay is not part of "
        "latency. Queueing inside the server still appears in TTFT."
    ),
    "output_tokens": (
        "Count of non-empty streamed pieces delivered to the client. This follows "
        "the provider's stream tokenization and is not a local retokenization."
    ),
    "error_rate": (
        "Failed measured requests, including timeouts, divided by measured requests "
        "issued. Warmup failures are stored and reported separately and are not in "
        "this rate."
    ),
    "percentile_method": (
        "Hyndman-Fan type 7: linear interpolation between the two nearest ranks. "
        "This matches NumPy's default percentile. With a small N, p95 and p99 "
        "describe this sample only."
    ),
}
