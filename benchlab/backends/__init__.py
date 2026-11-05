"""Serving backends.

``Backend`` is the adapter. Hugging Face Inference and the local transformers
HTTP server are implemented as clients. vLLM and TGI are thin streaming
clients against a server you run yourself.
"""

from benchlab.backends.base import Backend, BackendProtocolError
from benchlab.backends.registry import create_backend

__all__ = ["Backend", "BackendProtocolError", "create_backend"]
