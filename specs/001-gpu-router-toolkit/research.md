# Research: GPU Access Router Toolkit

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11
**Phase**: 0 — Pre-design research for all NEEDS CLARIFICATION items

---

## 1. Serial Request Queue (Python asyncio)

**Decision**: FastAPI + `asyncio.Queue` + `asyncio.Lock` + `asyncio.wait_for()`

**Rationale**: FastAPI's async endpoints accept concurrent requests naturally. A single `asyncio.Lock` around the Ollama inference call enforces one-at-a-time execution. `asyncio.wait_for(coro, timeout=300)` implements the per-request timeout in a Python 3.8-compatible way. A separate dict maps `request_id → queue_position` for O(1) position lookups by waiting clients.

**Key pattern**:
- Incoming requests each get a `request_id` (UUID) and initial position = `queue.qsize() + 1`
- A background asyncio task processes queue items serially under `asyncio.Lock`
- Waiting clients poll `GET /queue/{request_id}/position` to see their current position
- `asyncio.wait_for()` used instead of `asyncio.timeout()` (3.11+) for 3.8 compat
- No Janus needed: all server code is async (FastAPI); sync bridging deferred to future if needed

**Alternatives considered**:
- `threading.Queue` — rejected: blocks event loop, incompatible with async FastAPI
- `asyncio.Semaphore(1)` alone — rejected: cannot track queue position
- FastAPI background tasks — rejected: no built-in concurrency control or position tracking

---

## 2. Python Package Packaging

**Decision**: `pyproject.toml` + src-layout + Click CLI + lazy role imports

**Rationale**: Modern pyproject.toml packaging (setuptools backend) with `[project.optional-dependencies]` for `server`, `client`, `all` extras. src-layout (`src/gpu_access_router/`) prevents accidental import of local dev code and is 2025 best practice. Click chosen over Typer (more mature, no extra deps) and argparse (too verbose for nested sub-commands). Role-specific code loaded lazily inside Click command handlers so a client-only install never imports server dependencies.

**pyproject.toml key sections**:
```toml
[project]
name = "gpu-access-router"
requires-python = ">=3.8"
dependencies = ["click>=8.0", "tomli>=2.0; python_version<'3.11'", "tomli-w>=1.0"]

[project.optional-dependencies]
server = ["fastapi>=0.100", "uvicorn[standard]>=0.23", "ollama>=0.1", "rich>=13.0"]
client = ["ollama>=0.1", "rich>=13.0"]
all    = ["gpu-access-router[server,client]"]

[project.scripts]
gpu-access-router = "gpu_access_router.cli:main"
```

**Alternatives considered**:
- Typer — rejected for v1: adds dependency, less stable than Click; migration easy later
- Flat layout — rejected: src-layout is preferred for published packages
- argparse — rejected: verbose for 3-level nested commands (`server setup`, `config set`)

---

## 3. Ollama Integration

**Decision**: `ollama` Python SDK with remote `Client(host=)`, `/api/tags` for health + model discovery

**Key facts**:
- Remote client: `from ollama import Client; client = Client(host='http://100.x.x.x:11434')`
- `client.chat(model, messages, stream=False)` → `ChatResponse` with `.message.content`
- `client.list()` → list of available models (name, size, digest)
- `client.show('model-name')` → raises `ResponseError(status_code=404)` if not found
- Health check: `GET /api/tags` returns HTTP 200 when healthy (no dedicated `/health` endpoint)
- **Critical**: Server must set `OLLAMA_HOST=0.0.0.0:11434` env var in Docker to accept non-localhost connections

**Docker run command for server**:
```bash
docker run -d \
  --gpus all \
  --name ollama \
  -e OLLAMA_HOST=0.0.0.0:11434 \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --restart unless-stopped \
  ollama/ollama
```

**Alternatives considered**:
- Direct REST API via `httpx`/`requests` — rejected: Ollama Python SDK wraps this cleanly and is maintained by the Ollama project
- Streaming responses — deferred to v2 (out of scope)

---

## 4. Tailscale Connectivity Checking

**Decision**: `subprocess` + `tailscale status --json` + `socket.connect_ex()` for TCP probe

**Key patterns**:
```python
# Detect Tailscale installed
subprocess.run(['tailscale', '--version'], capture_output=True)  # FileNotFoundError if missing

# Check connected + get own IP
result = subprocess.run(['tailscale', 'status', '--json'], capture_output=True, text=True)
data = json.loads(result.stdout)
connected = data['BackendState'] == 'Running'
own_ip = data['Self']['TailscaleIPs'][0]  # IPv4

# TCP probe to server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(5)
reachable = sock.connect_ex((tailscale_ip, 11434)) == 0
```

**Ubuntu install**: `curl -fsSL https://tailscale.com/install.sh | sh`

**Alternatives considered**:
- ICMP ping — rejected: may be blocked by firewalls; TCP to Ollama port is more reliable
- Tailscale API — rejected: requires API token; CLI is available on all platforms

---

## 5. TOML Configuration (Python 3.8 compat)

**Decision**: `tomllib` (stdlib, 3.11+) with `tomli` fallback for 3.8–3.10, `tomli-w` for writes; `tomlkit` considered but rejected

**Import pattern**:
```python
try:
    import tomllib          # Python 3.11+ stdlib
except ImportError:
    import tomli as tomllib # pip install tomli for 3.8-3.10

import tomli_w              # pip install tomli-w (read only stdlib has no write)
```

**Config file location**: `~/.gpu-access-router/config.toml` (both roles share same file location, different `[server]`/`[client]` sections)

**Alternatives considered**:
- `tomlkit` — rejected: preserves formatting nicely but adds complexity; user-edited comments are not a priority for v1; can migrate later
- `configparser` (INI) — rejected: less expressive, no nested types, unfamiliar to modern Python devs
- Environment variables — rejected as primary mechanism (selected as Q2 answer was config file + CLI)

---

## Summary: Technology Stack

| Concern | Decision | Library/Tool |
|---|---|---|
| Web server (server role) | FastAPI + uvicorn | `fastapi`, `uvicorn[standard]` |
| Serial queue | asyncio.Queue + asyncio.Lock | stdlib asyncio |
| Per-request timeout | asyncio.wait_for() | stdlib asyncio |
| Ollama integration | Ollama Python SDK | `ollama` |
| CLI framework | Click nested groups | `click>=8.0` |
| Config format | TOML at ~/.gpu-access-router/config.toml | `tomli`/`tomllib` + `tomli-w` |
| Tailscale detection | subprocess + tailscale CLI | stdlib subprocess + socket |
| Output formatting | Rich tables/colors | `rich>=13.0` |
| Testing | pytest + pytest-asyncio | `pytest`, `pytest-asyncio` |
| Packaging | pyproject.toml + src-layout | setuptools |
