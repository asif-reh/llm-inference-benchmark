"""Workload configuration.

Paths inside a config file are resolved relative to that file, so a run does
not depend on the process working directory.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from benchlab.prompts import Prompt, load_prompts

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class ConfigError(ValueError):
    """The config file or the prompt set cannot be used."""


class InjectedError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Literal["warmup", "measure"]
    concurrency: int = Field(ge=1)
    batch_size: int = Field(default=1, ge=1)
    ordinal: int = Field(ge=0)
    error_type: Literal["timeout", "backend", "empty_stream"]
    message: str = "injected error"


class MockBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["mock"]
    model_id: str = "synthetic/mock-7b"
    revision: str | None = None
    injected_errors: list[InjectedError] = Field(default_factory=list)


class HuggingFaceBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["huggingface"]
    model_id: str
    revision: str | None = None
    provider: Literal["hf-inference", "auto"] = "hf-inference"
    mode: Literal["text", "chat"] = "chat"
    token_env: str = "HF_TOKEN"


class TransformersBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["transformers"]
    model_id: str
    revision: str | None = None
    base_url: str


class VllmBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["vllm"]
    model_id: str
    revision: str | None = None
    base_url: str
    api_key_env: str = "VLLM_API_KEY"


class TgiBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tgi"]
    model_id: str
    revision: str | None = None
    base_url: str


BackendConfig = Annotated[
    MockBackendConfig
    | HuggingFaceBackendConfig
    | TransformersBackendConfig
    | VllmBackendConfig
    | TgiBackendConfig,
    Field(discriminator="type"),
]


class WorkloadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: str
    concurrency: list[int] = Field(min_length=1)
    batch_sizes: list[int] = Field(default_factory=lambda: [1], min_length=1)
    warmup: int = Field(default=1, ge=0, le=10_000)
    samples: int = Field(ge=1, le=100_000)
    max_tokens: int = Field(ge=1, le=32_768)
    temperature: float = Field(default=0.0, ge=0)
    seed: int | None = 42
    timeout_s: float = Field(default=120, gt=0, le=3_600)

    @model_validator(mode="after")
    def _levels(self) -> WorkloadConfig:
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("concurrency levels must be unique")
        if any(item < 1 or item > 1024 for item in self.concurrency):
            raise ValueError("each concurrency level must be between 1 and 1024")
        if len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("batch_sizes must be unique")
        if any(item < 1 or item > 256 for item in self.batch_sizes):
            raise ValueError("each batch size must be between 1 and 256")
        return self


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: str
    overwrite: bool = False
    run_id: str | None = None
    freeze_created_at: str | None = None
    include_output_text: bool = True
    also_copy_summary_to: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids(self) -> OutputConfig:
        if self.run_id is not None and _NAME.match(self.run_id) is None:
            raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,80}")
        if self.freeze_created_at is not None:
            normalize_timestamp(self.freeze_created_at)
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_name: str
    clock: Literal["wall", "virtual"] = "wall"
    backend: BackendConfig
    workload: WorkloadConfig
    output: OutputConfig

    @model_validator(mode="after")
    def _relations(self) -> AppConfig:
        if _NAME.match(self.run_name) is None:
            raise ValueError("run_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,80}")
        if self.clock == "virtual" and self.backend.type != "mock":
            raise ValueError("clock: virtual is only supported for the mock backend")
        if self.backend.type not in {"mock", "transformers"} and any(
            size != 1 for size in self.workload.batch_sizes
        ):
            raise ValueError(
                f"backend '{self.backend.type}' does not fuse client-side batches. "
                "Set batch_sizes to [1] and apply load with concurrency. "
                "vLLM and TGI batch inside the server; configure that on the server, "
                "not in this client."
            )
        return self


class LoadedConfig:
    """Parsed config plus paths resolved relative to the config file."""

    def __init__(
        self,
        source: Path,
        config: AppConfig,
        prompts_path: Path,
        prompts: list[Prompt],
        output_dir: Path,
        copy_paths: list[Path],
    ) -> None:
        self.source = source
        self.config = config
        self.prompts_path = prompts_path
        self.prompts = prompts
        self.output_dir = output_dir
        self.copy_paths = copy_paths
        self._created_at: str | None = None

    @property
    def created_at(self) -> str:
        if self._created_at is None:
            frozen = self.config.output.freeze_created_at
            if frozen is not None:
                self._created_at = normalize_timestamp(frozen)
            else:
                self._created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._created_at

    @property
    def created_at_is_frozen(self) -> bool:
        return self.config.output.freeze_created_at is not None

    @property
    def run_id(self) -> str:
        chosen = self.config.output.run_id
        if chosen is not None:
            return chosen
        stamp = self.created_at.replace("-", "").replace(":", "")
        return f"{stamp}-{self.config.run_name}"


def normalize_timestamp(value: str) -> str:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"freeze_created_at is not an ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ValueError("freeze_created_at must include a timezone")
    return parsed.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_config(path: Path) -> LoadedConfig:
    source = path.expanduser()
    if not source.is_file():
        raise ConfigError(f"config not found: {source}")
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    try:
        config = AppConfig.model_validate(loaded)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    prompts_path = _resolve(source, config.workload.prompts)
    try:
        prompts = load_prompts(prompts_path)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    output_dir = _resolve(source, config.output.directory)
    copies = [_resolve(source, item) for item in config.output.also_copy_summary_to]
    return LoadedConfig(
        source=source.resolve(),
        config=config,
        prompts_path=prompts_path,
        prompts=prompts,
        output_dir=output_dir,
        copy_paths=copies,
    )


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()
