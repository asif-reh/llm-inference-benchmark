"""Construct a backend from a loaded config. Importing this module does not connect."""

from __future__ import annotations

from benchlab.backends.base import Backend
from benchlab.backends.hf_inference import HuggingFaceBackend
from benchlab.backends.mock import MockBackend
from benchlab.backends.tgi_client import TgiBackend
from benchlab.backends.transformers_local import TransformersBackend
from benchlab.backends.vllm_client import VllmBackend
from benchlab.config import (
    HuggingFaceBackendConfig,
    LoadedConfig,
    MockBackendConfig,
    TgiBackendConfig,
    TransformersBackendConfig,
    VllmBackendConfig,
)


def create_backend(loaded: LoadedConfig) -> Backend:
    backend = loaded.config.backend
    timeout_s = loaded.config.workload.timeout_s
    seed = loaded.config.workload.seed
    if isinstance(backend, MockBackendConfig):
        return MockBackend(backend, seed=0 if seed is None else seed)
    if isinstance(backend, HuggingFaceBackendConfig):
        return HuggingFaceBackend(backend, seed=seed, timeout_s=timeout_s)
    if isinstance(backend, TransformersBackendConfig):
        return TransformersBackend(
            model_id=backend.model_id,
            revision=backend.revision,
            base_url=backend.base_url,
            timeout_s=timeout_s,
        )
    if isinstance(backend, VllmBackendConfig):
        return VllmBackend(
            model_id=backend.model_id,
            revision=backend.revision,
            base_url=backend.base_url,
            api_key_env=backend.api_key_env,
            timeout_s=timeout_s,
            seed=seed,
        )
    if isinstance(backend, TgiBackendConfig):
        return TgiBackend(
            model_id=backend.model_id,
            revision=backend.revision,
            base_url=backend.base_url,
            timeout_s=timeout_s,
            seed=seed,
        )
    raise TypeError(f"unsupported backend config: {type(backend).__name__}")
