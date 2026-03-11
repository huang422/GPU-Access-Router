"""GPU Directer — route LLM inference to a remote GPU server or local Ollama."""

__version__ = "0.1.0"

from gpu_directer.core.exceptions import (  # noqa: F401
    GPUDirecterError,
    GPUDirecterConfigError,
    GPUDirecterConnectionError,
    GPUDirecterTimeoutError,
)

try:
    from gpu_directer.client.router import GPURouter  # noqa: F401
except ImportError:
    pass  # ollama not installed; GPURouter requires [client] extra
