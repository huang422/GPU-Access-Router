class GPUAccessRouterError(Exception):
    """Base exception for all GPU Access Router errors."""


class GPUAccessRouterConfigError(GPUAccessRouterError):
    """Configuration file is missing, malformed, or contains invalid values."""


class GPUAccessRouterConnectionError(GPUAccessRouterError):
    """No routing target is available (remote and/or local Ollama unreachable)."""


class GPUAccessRouterTimeoutError(GPUAccessRouterError):
    """A request waited in the server queue longer than the configured timeout."""
