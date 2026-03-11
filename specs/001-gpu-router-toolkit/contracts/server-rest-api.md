# Contract: Server REST API

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11
**Base URL**: `http://<tailscale-ip>:11434` (same port as Ollama; GPU Directer server wraps Ollama)

The GPU Directer server adds a thin FastAPI layer on top of Ollama that enforces the serial queue. All Ollama-native endpoints remain accessible; the server adds `/gd/*` prefixed routes for queue management and health.

---

## Health & Status

### `GET /gd/health`

Returns server health and queue status.

**Response 200**:
```json
{
  "status": "ok",
  "queue_depth": 2,
  "processing": true,
  "ollama_reachable": true,
  "gpu_available": true,
  "uptime_seconds": 3600
}
```

**Response 503** (Ollama unreachable):
```json
{
  "status": "degraded",
  "error": "Ollama container not responding",
  "queue_depth": 0
}
```

---

## Inference (Queued)

### `POST /gd/chat`

Submit an inference request to the serial queue. Returns immediately with a `request_id` and queue position. The client then polls for completion.

**Request body**:
```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "options": {},
  "timeout": 300
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | yes | — | Ollama model name |
| `messages` | yes | — | Chat message array |
| `options` | no | `{}` | Ollama generation options (temperature, etc.) |
| `timeout` | no | 300 | Max seconds to wait in queue |

**Response 202 Accepted**:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "queue_position": 3,
  "estimated_wait_seconds": 120,
  "status": "waiting"
}
```

**Response 503** (queue full):
```json
{
  "error": "Queue full",
  "max_depth": 10,
  "current_depth": 10
}
```

---

### `GET /gd/queue/{request_id}`

Poll for the status of a queued request.

**Response 200 — waiting**:
```json
{
  "request_id": "550e8400-...",
  "status": "waiting",
  "queue_position": 2,
  "queued_at": "2026-03-11T12:00:00Z"
}
```

**Response 200 — processing**:
```json
{
  "request_id": "550e8400-...",
  "status": "processing",
  "queue_position": 0,
  "started_at": "2026-03-11T12:00:05Z"
}
```

**Response 200 — complete**:
```json
{
  "request_id": "550e8400-...",
  "status": "complete",
  "result": {
    "model": "llama3.2",
    "created_at": "2026-03-11T12:00:15Z",
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you?"
    },
    "done": true
  }
}
```

**Response 200 — timeout**:
```json
{
  "request_id": "550e8400-...",
  "status": "timeout",
  "error": "Request exceeded queue timeout of 300 seconds",
  "timeout_seconds": 300
}
```

**Response 404** (unknown request_id):
```json
{
  "error": "Request not found",
  "request_id": "550e8400-..."
}
```

---

## Models

### `GET /gd/models`

List all models available on this server's Ollama instance. Proxies to Ollama `/api/tags`.

**Response 200**:
```json
{
  "models": [
    {
      "name": "llama3.2",
      "size": 3800000000,
      "digest": "sha256:...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "3.2B",
        "quantization_level": "Q4_K_M"
      }
    }
  ]
}
```

---

## Queue Management

### `GET /gd/queue`

Returns current queue state.

**Response 200**:
```json
{
  "depth": 2,
  "max_depth": 10,
  "processing": true,
  "requests": [
    {
      "request_id": "550e8400-...",
      "position": 1,
      "model": "llama3.2",
      "queued_at": "2026-03-11T12:00:00Z"
    }
  ]
}
```

---

## Notes

- The `GPURouter` Python client handles the submit-then-poll pattern transparently. Developers calling `router.chat()` never interact with the REST API directly.
- The Ollama native endpoints (`/api/chat`, `/api/tags`, etc.) remain accessible directly on the same port for advanced users, but they **bypass the queue** — only `/gd/chat` enforces serial execution.
- All timestamps are ISO 8601 UTC.
- All error responses include an `"error"` string field.
