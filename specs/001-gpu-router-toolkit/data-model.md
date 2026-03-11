# Data Model: GPU Access Router Toolkit

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11
**Source**: spec.md entities + research.md technology decisions

---

## Entities

### 1. Configuration (TOML file: `~/.gpu-access-router/config.toml`)

The on-disk representation of all user-configurable settings. Both roles (server and client) use the same file path but different top-level sections. Written by setup wizards; read by all runtime components.

**File structure**:
```toml
[client]
server_ip       = "100.x.x.x"          # Tailscale IP of the GPU server
server_port     = 11434                  # Ollama port on server (default: 11434)
routing_mode    = "auto"                 # "auto" | "remote" | "local"
timeout_seconds = 300                    # Queue wait timeout in seconds
default_model   = ""                     # Optional: pre-select a model

[server]
ollama_port     = 11434                  # Port Ollama container listens on
queue_timeout   = 300                    # Max seconds a request waits in queue
max_queue_depth = 10                     # Max requests allowed in queue (0 = unlimited)

[meta]
role            = "client"               # "client" | "server" | "both"
version         = "0.1.0"               # Config schema version
created_at      = "2026-03-11T00:00:00" # ISO 8601 timestamp
```

**Validation rules**:
- `routing_mode` must be one of `"auto"`, `"remote"`, `"local"`
- `timeout_seconds` and `queue_timeout` must be positive integers
- `server_ip` must be a valid IPv4 address when `routing_mode != "local"`
- `server_port` and `ollama_port` must be in range 1–65535

**State transitions**: Configuration is created by setup wizard → updated via `config set` or direct file edit → read on every `GPURouter` call.

---

### 2. GPURouter (Python class, client-side runtime)

The primary developer-facing object. Instantiated once per project, called for each inference. Reads configuration at instantiation time, evaluates routing logic per call.

**Attributes**:
| Field | Type | Description |
|---|---|---|
| `routing_mode` | `str` | `"auto"` / `"remote"` / `"local"` — from config, overridable per call |
| `server_ip` | `str \| None` | Tailscale IP of remote server from config |
| `server_port` | `int` | Remote Ollama port (default 11434) |
| `timeout` | `int` | Default request timeout in seconds |
| `_remote_client` | `ollama.Client \| None` | Cached remote Ollama client |
| `_local_client` | `ollama.Client \| None` | Cached local Ollama client |

**Methods**:
| Method | Signature | Description |
|---|---|---|
| `chat` | `(model: str, messages: list, prefer: str = None, timeout: int = None) → ChatResponse` | Route and execute chat inference |
| `list_models` | `(source: str = "auto") → dict` | List available models on remote/local/both |
| `status` | `() → dict` | Check connectivity to all configured sources |

**Routing logic** (for `prefer="auto"` or config `routing_mode="auto"`):
1. Check if remote server is reachable (TCP probe to `server_ip:server_port`)
2. If reachable: check if requested model exists on remote via `/api/tags`
3. If model found on remote: route to remote, return response
4. If model NOT found on remote but exists locally: fall back to local, emit warning
5. If remote unreachable: fall back to local silently
6. If local also unavailable: raise `GPUAccessRouterError` with details of what was tried

---

### 3. InferenceRequest (server-side runtime, not persisted)

Represents a single queued or in-flight inference request on the server. Lives only in memory; not persisted to disk.

**Fields**:
| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | UUID assigned at queue entry |
| `model` | `str` | Requested Ollama model name |
| `messages` | `list[dict]` | Chat message list |
| `queued_at` | `float` | Unix timestamp when request entered queue |
| `started_at` | `float \| None` | Unix timestamp when inference began |
| `status` | `str` | `"waiting"` / `"processing"` / `"complete"` / `"timeout"` / `"error"` |
| `queue_position` | `int` | Current position in queue (1 = next to run) |
| `result` | `dict \| None` | Inference response once complete |
| `error` | `str \| None` | Error message if status is error/timeout |

**State transitions**:
```
→ waiting (queued_at set, position assigned)
     ↓ (lock acquired, all requests ahead complete)
→ processing (started_at set, Ollama call begins)
     ↓ (Ollama returns)
→ complete (result set)
     OR
     ↓ (asyncio.wait_for timeout fires)
→ timeout (error set, queue slot released)
     OR
     ↓ (Ollama error)
→ error (error set)
```

---

### 4. RequestQueue (server-side runtime, singleton)

The in-memory serial queue managing all inference requests on the server. One instance per server process.

**Fields**:
| Field | Type | Description |
|---|---|---|
| `_queue` | `asyncio.Queue` | Underlying async queue of pending requests |
| `_lock` | `asyncio.Lock` | Ensures one inference runs at a time |
| `_pending` | `dict[str, InferenceRequest]` | request_id → InferenceRequest for position lookup |
| `timeout_seconds` | `int` | Default queue wait timeout (from config, default 300) |

**Operations**:
| Operation | Description |
|---|---|
| `enqueue(request)` | Add request to queue, return `{request_id, position}` |
| `get_position(request_id)` | Return current queue position (1-indexed) |
| `get_depth()` | Return total number of waiting + processing requests |
| `process_loop()` | Background asyncio task: pulls from queue, runs inference under lock |

---

### 5. DiagnosticReport (server-side, generated on demand)

Produced by `gpu-access-router server doctor`. Represents the result of all extended health checks.

**Structure**:
```python
{
  "timestamp": "2026-03-11T00:00:00Z",
  "overall": "pass" | "fail",
  "checks": [
    {
      "name": "docker_installed",
      "status": "pass" | "fail",
      "detail": "Docker 24.0.7",
      "fix_hint": ""  # empty string if passed
    },
    {
      "name": "ollama_container_running",
      "status": "pass" | "fail",
      "detail": "Container 'ollama' is running",
      "fix_hint": "Run: docker start ollama"
    },
    {
      "name": "gpu_passthrough",
      "status": "pass" | "fail",
      "detail": "NVIDIA GeForce RTX 4090 detected",
      "fix_hint": "Install nvidia-container-toolkit: https://..."
    },
    {
      "name": "tailscale_connected",
      "status": "pass" | "fail",
      "detail": "Connected. Tailscale IP: 100.x.x.x",
      "fix_hint": "Run: sudo tailscale up"
    },
    {
      "name": "ollama_models_available",
      "status": "pass" | "fail",
      "detail": "3 models: llama3.2, mistral, codellama",
      "fix_hint": "Pull a model: docker exec ollama ollama pull llama3.2"
    },
    {
      "name": "queue_status",
      "status": "pass",
      "detail": "Queue depth: 0, Processing: idle",
      "fix_hint": ""
    }
  ]
}
```

---

## Relationships

```
Configuration (file)
    ↑ reads
GPURouter ──────────────→ InferenceRequest (submitted to remote)
    ↓ TCP probe                ↓
Remote Server API      RequestQueue (server-side, holds InferenceRequests)
    ↓                          ↓ serially processes
Ollama Docker Container ←── asyncio.Lock (enforces one at a time)

DiagnosticReport ← server doctor (reads Docker, Tailscale, Ollama state)
```

---

## Data Volume & Scale Assumptions

- Config file: <1 KB; single file per machine; read on each `GPURouter` instantiation
- RequestQueue: max 10 items by default (configurable); each InferenceRequest ~1 KB in memory
- DiagnosticReport: generated on demand, never persisted; ~500 bytes
- Ollama model list: typically 1–20 models; fetched fresh on each `list_models()` call
- No database required; all persistence is the config TOML file + Ollama's own model storage
