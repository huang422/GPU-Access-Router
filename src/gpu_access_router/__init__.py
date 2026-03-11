"""GPU Access Router — route LLM inference to a remote GPU server or local Ollama."""

__version__ = "0.1.0"

from gpu_access_router.core.exceptions import (  # noqa: F401
    GPUAccessRouterError,
    GPUAccessRouterConfigError,
    GPUAccessRouterConnectionError,
    GPUAccessRouterTimeoutError,
)

try:
    from gpu_access_router.client.router import GPURouter  # noqa: F401
except ImportError:
    pass  # ollama not installed; GPURouter requires [client] extra
