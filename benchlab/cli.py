"""Command line for benchlab.

``run`` executes a config. ``validate`` loads it and the prompt file and stops.
``--dry-run`` prints the plan and does not construct a backend, so it does not
need a network or an API token.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from benchlab import __version__
from benchlab.backends.registry import create_backend
from benchlab.clock import Clock, RealClock, VirtualClock, run_with_clock
from benchlab.config import ConfigError, LoadedConfig, load_config
from benchlab.export import write_run
from benchlab.prompts import file_sha256
from benchlab.runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchlab",
        description=(
            "Benchmark LLM serving configurations from the client. "
            "Warmup is excluded from reported statistics."
        ),
    )
    parser.add_argument("--version", action="version", version=f"benchlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="load a config and its prompt file")
    validate.add_argument("--config", required=True, type=Path)

    run = sub.add_parser("run", help="execute a benchmark config")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without sending requests",
    )
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override the output directory in the config",
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args.config)
        if args.command == "run":
            return _run(args.config, dry_run=bool(args.dry_run), output=args.output)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _validate(path: Path) -> int:
    loaded = load_config(path)
    print(_plan(loaded))
    print("valid")
    return 0


def _run(path: Path, *, dry_run: bool, output: Path | None) -> int:
    loaded = load_config(path)
    if output is not None:
        loaded.output_dir = output if output.is_absolute() else (Path.cwd() / output).resolve()
    print(_plan(loaded))
    if dry_run:
        print("dry-run: no requests sent")
        return 0

    backend = create_backend(loaded)

    async def _execute(clock: Clock) -> int:
        summary, records = await run_benchmark(
            loaded, backend, clock, log=lambda line: print(line, flush=True)
        )
        write_run(
            loaded.output_dir,
            summary,
            records,
            copies=loaded.copy_paths,
            overwrite=loaded.config.output.overwrite,
            include_output_text=loaded.config.output.include_output_text,
        )
        print(f"wrote {loaded.output_dir / 'summary.json'}")
        print(f"wrote {loaded.output_dir / 'summary.csv'}")
        print(f"wrote {loaded.output_dir / 'requests.jsonl'}")
        for copy in loaded.copy_paths:
            print(f"copied summary to {copy}")
        notice = summary.get("notice")
        if isinstance(notice, str):
            print(notice)
        return 0

    if loaded.config.clock == "virtual":
        return asyncio.run(run_with_clock(VirtualClock(), _execute))
    return asyncio.run(_execute(RealClock()))


def _plan(loaded: LoadedConfig) -> str:
    config = loaded.config
    backend = config.backend
    digest = file_sha256(loaded.prompts_path)
    lines = [
        f"run_id: {loaded.run_id}",
        f"run_name: {config.run_name}",
        f"backend: {backend.type}",
        f"model_id: {backend.model_id}",
        f"revision: {backend.revision if backend.revision is not None else '(unpinned)'}",
        f"clock: {config.clock}",
        f"prompts: {len(loaded.prompts)} from {config.workload.prompts} sha256={digest[:12]}",
        f"concurrency: {config.workload.concurrency}",
        f"batch_sizes: {config.workload.batch_sizes}",
        f"warmup_per_level: {config.workload.warmup} (excluded from stats)",
        f"samples_per_level: {config.workload.samples}",
        f"max_tokens: {config.workload.max_tokens}",
        f"temperature: {config.workload.temperature}",
        f"seed: {config.workload.seed}",
        f"timeout_s: {config.workload.timeout_s}",
        f"output: {loaded.output_dir}",
    ]
    return "\n".join(lines)
