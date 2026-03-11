# Implementation Plan: GPU Directer Toolkit

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-gpu-router-toolkit/spec.md`

---

## Summary

GPU Directer is a two-role pip-installable Python toolkit that routes LLM inference calls to a remote GPU server (gpu-server running Ollama in Docker over Tailscale) or falls back to local Ollama automatically. The server role runs a FastAPI app with an asyncio serial queue (one inference at a time) to prevent GPU OOM. The client role provides a `GPURouter` Python class and a `gpu-directer` Click CLI. Configuration is stored in `~/.gpu-directer/config.toml` (TOML format, editable directly or via CLI). Both roles are installable from GitHub via `pip install git+https://...[server|client|all]`.

---

## Technical Context

**Language/Version**: Python 3.8+ (supports 3.8–3.12+)
**Primary Dependencies**:
- Core (always): `click>=8.0`, `tomli>=2.0` (Python <3.11 only), `tomli-w>=1.0`, `rich>=13.0`
- Server extras: `fastapi>=0.100`, `uvicorn[standard]>=0.23`, `ollama>=0.1`
- Client extras: `ollama>=0.1`

**Storage**: `~/.gpu-directer/config.toml` — flat TOML file, ~1 KB, no database

**Testing**: `pytest` + `pytest-asyncio` (for async queue tests)

**Target Platform**:
- Server: Ubuntu 20.04+ with NVIDIA GPU + Docker
- Client: macOS 12+ or Ubuntu 20.04+

**Project Type**: Python library + CLI tool (src-layout, pip-installable from GitHub)

**Performance Goals**:
- Server health endpoint responds in < 2 seconds
- Client setup wizard completes in < 60 seconds
- Queue depth visible to clients within 1 polling cycle (< 1 second delay)

**Constraints**:
- Python 3.8+ compatibility (no `asyncio.timeout()`, no `asyncio.TaskGroup()`, no `tomllib` stdlib — use `tomli` fallback)
- No root/sudo required on client machines
- Single `pip install` with no mandatory manual steps

**Scale/Scope**: Personal use — 1 GPU server, 1–5 client machines, queue depth ≤ 10 concurrent requests

---

## Constitution Check

The project constitution (`/.specify/memory/constitution.md`) contains only placeholder template content — no project-specific principles have been ratified. No constitution gates apply to this feature.

*Re-check: Not applicable (constitution unpopulated).*

---

## Project Structure

### Documentation (this feature)

```text
specs/001-gpu-router-toolkit/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: technology decisions
├── data-model.md        # Phase 1: entities and data shapes
├── quickstart.md        # Phase 1: end-to-end user guide
├── contracts/
│   ├── cli-schema.md        # Full CLI command surface
│   ├── python-api.md        # GPURouter public Python API
│   └── server-rest-api.md   # Server FastAPI REST endpoints
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/
└── gpu_directer/
    ├── __init__.py              # Public exports: GPURouter, exceptions
    ├── __main__.py              # python -m gpu_directer support
    ├── cli.py                   # Click entry point (all sub-command groups)
    ├── config.py                # TOML config read/write + validation
    ├── core/
    │   ├── __init__.py
    │   ├── exceptions.py        # GPUDirecterError, ConnectionError, TimeoutError, ConfigError
    │   └── constants.py         # Default port, timeout, config path, etc.
    ├── server/
    │   ├── __init__.py          # Lazy import guard (requires [server] extra)
    │   ├── api.py               # FastAPI app: /gd/health, /gd/chat, /gd/queue, /gd/models
    │   ├── queue.py             # SerialQueue: asyncio.Queue + asyncio.Lock + asyncio.wait_for
    │   ├── doctor.py            # Diagnostic checks: Docker, Ollama, GPU, Tailscale, models
    │   └── setup_wizard.py      # Interactive server setup flow
    └── client/
        ├── __init__.py          # Lazy import guard (requires [client] extra)
        ├── router.py            # GPURouter class: routing logic + fallback
        ├── setup_wizard.py      # Interactive client setup flow
        └── status.py            # Status display for client status command

pyproject.toml                   # Package config: extras, entry points, metadata
tests/
├── unit/
│   ├── test_config.py           # Config read/write/validation
│   ├── test_router.py           # GPURouter routing logic (mocked Ollama)
│   └── test_queue.py            # Serial queue + timeout behavior (async)
└── integration/
    ├── test_server_api.py       # FastAPI endpoints with TestClient
    └── test_cli.py              # CLI commands with Click testing utilities
```

**Structure Decision**: Single project (Option 1), src-layout. No frontend. No database. Server and client share one package with role-specific extras isolating heavy dependencies.

---

## Complexity Tracking

No constitution violations to justify.

---

## Implementation Phases

### Phase 1: Foundation (core + config + packaging skeleton)

**Deliverables**: `pyproject.toml`, `core/exceptions.py`, `core/constants.py`, `config.py`, basic `cli.py` skeleton with all command groups registered (no implementations), `gpu-directer --help` works after install.

**Why first**: Everything else depends on config reading and the CLI skeleton. No role-specific deps needed yet.

---

### Phase 2: Server — Serial Queue + FastAPI

**Deliverables**: `server/queue.py` (SerialQueue with asyncio.Queue + Lock + wait_for timeout), `server/api.py` (FastAPI app: `/gd/health`, `/gd/chat`, `/gd/queue/{id}`, `/gd/models`), unit tests for queue behavior (serial enforcement, timeout, position tracking).

**Key design**: `POST /gd/chat` returns 202 immediately with `request_id` + position. Background asyncio task processes queue serially. Client polls `GET /gd/queue/{id}` until `status == "complete"` or `"timeout"`.

---

### Phase 3: Server — Setup Wizard + Doctor

**Deliverables**: `server/setup_wizard.py` (checks Docker → NVIDIA drivers → pull ollama/ollama → start container with `--gpus all -e OLLAMA_HOST=0.0.0.0` → install Tailscale → print IP), `server/doctor.py` (6 checks: Docker, container running, GPU passthrough, Tailscale, models available, queue status), `gpu-directer server setup` and `gpu-directer server doctor` CLI commands wired up.

---

### Phase 4: Client — GPURouter

**Deliverables**: `client/router.py` (GPURouter with auto/remote/local routing, fallback logic, model-missing warning, submit-then-poll pattern for `/gd/chat`), unit tests for all routing cases (remote up, remote down, model missing, both down), `from gpu_directer import GPURouter` works.

---

### Phase 5: Client — Setup Wizard + Status

**Deliverables**: `client/setup_wizard.py` (Tailscale check → TCP probe → `/gd/models` query → mode prompt → write config), `client/status.py` (display server online/offline, queue depth, models, local Ollama status), `gpu-directer client setup` and `gpu-directer client status` CLI commands wired up.

---

### Phase 6: Config CLI Commands

**Deliverables**: `gpu-directer config show` (pretty-print config.toml), `gpu-directer config set k=v` (validate key + type, write), `gpu-directer config edit` (open in $EDITOR), `gpu-directer config reset` (confirm + overwrite defaults).

---

### Phase 7: Polish + Documentation

**Deliverables**: README with Tailscale setup guide, server quick-start, client quick-start, `GPURouter` code example, complete command reference. Integration tests. `--json` flag on all commands. `--quiet` flag. `gpu-directer --version`.

---

## Key Technical Decisions (from research.md)

| Decision | Choice | Reason |
|---|---|---|
| Queue implementation | asyncio.Queue + asyncio.Lock + asyncio.wait_for | Python 3.8 compat; no Janus needed (all server code is async) |
| Web framework | FastAPI + uvicorn | Async-native, automatic OpenAPI, production-ready |
| CLI framework | Click | Mature, nested groups, no extra deps vs Typer |
| Config format | TOML via tomli/tomllib + tomli-w | Human-readable, Python 3.8 compat with tomli fallback |
| Ollama integration | ollama Python SDK + `Client(host=)` | Official SDK, same return type as local calls |
| Tailscale detection | subprocess + `tailscale status --json` | Works on all platforms; no Tailscale API token needed |
| Package structure | src-layout + optional-dependencies extras | 2025 best practice; clean role separation |

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Ollama container needs `OLLAMA_HOST=0.0.0.0` or it rejects remote connections | Setup wizard sets this env var; doctor check verifies Ollama responds on Tailscale IP |
| Queue bypassed by direct Ollama API calls | Document clearly: only `/gd/chat` is queued; direct `/api/chat` bypasses queue intentionally for power users |
| asyncio.wait_for cancels the wrong task in nested awaits | Wrap only the Ollama HTTP call, not the queue entry, in wait_for |
| Tailscale IP changes after mobile hotspot reconnect | Client `status` command probes on every call; config update via `config set` is documented in troubleshooting |
| Python 3.8 tomllib not in stdlib | `try: import tomllib except ImportError: import tomli as tomllib` pattern covers all versions |
