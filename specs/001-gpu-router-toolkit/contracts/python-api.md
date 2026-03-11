# Contract: Python API (GPURouter)

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11
**Module**: `gpu_access_router` (installed via `pip install gpu-access-router[client]`)

This document defines the public Python API contract for `GPURouter` — the primary developer-facing class.

---

## Import

```python
from gpu_access_router import GPURouter
```

---

## `GPURouter`

### Constructor

```python
GPURouter(
    config_path: str = None,    # Path to config.toml. Default: ~/.gpu-access-router/config.toml
    routing_mode: str = None,   # Override config routing_mode for this instance
    timeout: int = None,        # Override config timeout_seconds for this instance
)
```

Reads configuration at instantiation. Raises `GPUAccessRouterConfigError` if config file is missing or malformed.

---

### `GPURouter.chat()`

```python
def chat(
    model: str,                     # Ollama model name, e.g. "llama3.2"
    messages: list[dict],           # Ollama-format message list
    prefer: str = None,             # Per-call routing override: "auto"|"remote"|"local"
    timeout: int = None,            # Per-call timeout override in seconds
    **kwargs,                       # Passed through to Ollama client (options, format, etc.)
) -> ChatResponse
```

**Returns**: `ollama.ChatResponse` — same object returned by `ollama.Client.chat()`. Access response text via `.message.content`.

**Routing behavior** (when `prefer=None`, uses instance/config `routing_mode`):
- `"remote"` — route to remote server only; raise `GPUAccessRouterConnectionError` if unreachable
- `"local"` — route to local Ollama only; raise `GPUAccessRouterConnectionError` if unavailable
- `"auto"` (default):
  1. If remote reachable AND model exists on remote → route remote
  2. If remote reachable BUT model missing on remote AND model exists locally → route local, emit `UserWarning`
  3. If remote unreachable AND local available → route local silently
  4. If both unavailable → raise `GPUAccessRouterConnectionError`

**Exceptions**:
| Exception | When raised |
|---|---|
| `GPUAccessRouterConnectionError` | No available routing target found |
| `GPUAccessRouterTimeoutError` | Request waited in server queue longer than `timeout` seconds |
| `GPUAccessRouterConfigError` | Config file missing, malformed, or has invalid values |
| `ollama.ResponseError` | Ollama returned an error (model error, generation error, etc.) |

**Example**:
```python
from gpu_access_router import GPURouter

router = GPURouter()

response = router.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.message.content)

# Force local for this call only
response = router.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Debug this locally"}],
    prefer="local"
)
```

---

### `GPURouter.list_models()`

```python
def list_models(
    source: str = "auto",   # "remote" | "local" | "auto" (both)
) -> dict
```

**Returns**:
```python
{
    "remote": [
        {"name": "llama3.2", "size": 3800000000, "details": {...}},
        ...
    ],
    "local": [
        {"name": "llama3.2", "size": 3800000000, "details": {...}},
    ],
    "reachable": {
        "remote": True,
        "local": True
    }
}
```

If a source is unreachable, its list is `None` and `reachable[source]` is `False`. No exception raised.

---

### `GPURouter.status()`

```python
def status() -> dict
```

**Returns**:
```python
{
    "remote": {
        "reachable": True,
        "server_ip": "100.64.0.5",
        "port": 11434,
        "queue_depth": 2,
        "models": ["llama3.2", "mistral"]
    },
    "local": {
        "reachable": True,
        "models": ["llama3.2"]
    },
    "routing_mode": "auto",
    "config_path": "/Users/user/.gpu-access-router/config.toml"
}
```

---

## Exceptions Module

```python
from gpu_access_router.exceptions import (
    GPUAccessRouterError,           # Base exception
    GPUAccessRouterConfigError,     # Config missing/invalid
    GPUAccessRouterConnectionError, # No routing target available
    GPUAccessRouterTimeoutError,    # Queue wait timeout exceeded
)
```

All exceptions inherit from `GPUAccessRouterError` for easy `except` catching.

---

## Warnings

`GPURouter.chat()` uses Python's `warnings.warn()` for non-fatal routing events:

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    response = router.chat("llama3.2", messages)
    # w[0].message will contain routing warning if emitted
```

Standard Python warning filter applies: warnings shown once per location by default.

---

## Compatibility

- Returns `ollama.ChatResponse` objects — identical to calling `ollama.Client.chat()` directly, so existing code using the Ollama SDK can drop in `GPURouter` with minimal changes.
- `**kwargs` in `chat()` are forwarded to Ollama, supporting `options={"temperature": 0.7}`, `format="json"`, etc.
