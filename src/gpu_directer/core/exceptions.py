class GPUDirecterError(Exception):
    """Base exception for all GPU Directer errors."""


class GPUDirecterConfigError(GPUDirecterError):
    """Configuration file is missing, malformed, or contains invalid values."""


class GPUDirecterConnectionError(GPUDirecterError):
    """No routing target is available (remote and/or local Ollama unreachable)."""


class GPUDirecterTimeoutError(GPUDirecterError):
    """A request waited in the server queue longer than the configured timeout."""
