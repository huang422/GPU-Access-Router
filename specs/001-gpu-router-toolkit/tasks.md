---
description: "Task list for GPU Directer Toolkit implementation"
---

# Tasks: GPU Directer Toolkit

**Input**: Design documents from `/specs/001-gpu-router-toolkit/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

**Organization**: Tasks grouped by user story. No test tasks (TDD not requested in spec).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependencies)
- **[Story]**: User story label (US1–US6) maps to spec.md priorities
- Paths assume src-layout: `src/gpu_directer/`, `tests/` at repo root

---

## Phase 1: Setup (Project Skeleton)

**Purpose**: Project structure, packaging config, and shared core modules. No role-specific dependencies.

- [X] T001 Create directory tree: `src/gpu_directer/{core,server,client}/`, `tests/{unit,integration}/` with `__init__.py` stubs
- [X] T002 Create `pyproject.toml` with metadata, `requires-python = ">=3.8"`, core deps (`click>=8.0`, `tomli>=2.0; python_version<"3.11"`, `tomli-w>=1.0`, `rich>=13.0`), optional extras `[server]` (`fastapi>=0.100`, `uvicorn[standard]>=0.23`, `ollama>=0.1`), `[client]` (`ollama>=0.1`), `[all]` (`gpu-directer[server,client]`), console script `gpu-directer = "gpu_directer.cli:main"`, and setuptools src-layout config
- [X] T003 [P] Create `src/gpu_directer/core/constants.py` defining `DEFAULT_PORT = 11434`, `DEFAULT_TIMEOUT = 300`, `DEFAULT_QUEUE_DEPTH = 10`, `DEFAULT_ROUTING_MODE = "auto"`, `CONFIG_PATH = Path.home() / ".gpu-directer" / "config.toml"`
- [X] T004 [P] Create `src/gpu_directer/core/exceptions.py` defining `GPUDirecterError` (base), `GPUDirecterConfigError`, `GPUDirecterConnectionError`, `GPUDirecterTimeoutError` — all inheriting from `GPUDirecterError`
- [X] T005 Create `src/gpu_directer/__init__.py` with placeholder exports: `from gpu_directer.core.exceptions import *` and `__version__ = "0.1.0"`
- [X] T006 Create `src/gpu_directer/__main__.py` with `from gpu_directer.cli import main; main()` for `python -m gpu_directer` support

**Checkpoint**: `pip install -e .` succeeds; package importable; no CLI yet

---

## Phase 2: Foundation (Config + CLI Skeleton)

**Purpose**: Config TOML read/write and the full CLI command tree registered. Every command must exist (even if stubbed), so `gpu-directer --help` shows all sub-commands.

**⚠️ CRITICAL**: No user story implementation can begin until config.py and cli.py skeleton are complete

- [X] T007 Implement `src/gpu_directer/config.py`: `load_config(path=None) → dict` (reads TOML, creates default if missing), `save_config(data, path=None)` (writes TOML), `get(key, default=None)` (dotted key access e.g. `"client.server_ip"`), `set_value(key, value)` (dotted key write with type validation), `validate_config(data)` raising `GPUDirecterConfigError` on invalid `routing_mode`, negative timeouts, or invalid port range. Default config structure matches data-model.md TOML schema.
- [X] T008 [P] Create `src/gpu_directer/server/__init__.py` with lazy import guard: `try: import fastapi; HAVE_SERVER = True except ImportError: HAVE_SERVER = False`
- [X] T009 [P] Create `src/gpu_directer/client/__init__.py` with lazy import guard: `try: import ollama; HAVE_CLIENT = True except ImportError: HAVE_CLIENT = False`
- [X] T010 Implement full CLI skeleton in `src/gpu_directer/cli.py` using Click: top-level `@click.group main` with `--version` flag; `server` group with stubbed `setup`, `doctor`, `models`, `start`, `stop`, `restart` commands; `client` group with stubbed `setup`, `status`, `models` commands; `config` group with stubbed `show`, `set`, `edit`, `reset` commands. Each stub prints "Not yet implemented" and exits 0. Global `--config PATH` option on `main` group. Each command accepts `--json` and `--quiet` flags.

**Checkpoint**: `pip install -e .` + `gpu-directer --help` shows all command groups; `gpu-directer server --help` lists all server sub-commands; config.py unit-testable in isolation

---

## Phase 3: US1 — Server Setup: Install and Configure GPU Server (Priority: P1) 🎯 MVP

**Goal**: A user on an Ubuntu+NVIDIA machine runs `gpu-directer server setup` and ends with a running Ollama Docker container with GPU access, Tailscale connected, and the server printing its Tailscale IP. `gpu-directer server doctor` reports all checks pass.

**Independent Test**: On a single Ubuntu+NVIDIA machine: run `gpu-directer server setup`, then `gpu-directer server doctor` — all 6 checks pass; `curl http://localhost:11434/api/tags` returns HTTP 200.

### Server Queue + API (prerequisite for server to accept requests)

- [X] T011 Implement `src/gpu_directer/server/queue.py`: `SerialQueue` class with `__init__(timeout_seconds=300, max_depth=10)`, `async enqueue(request_data: dict) → dict` (returns `{request_id, queue_position, status}`), `async get_status(request_id: str) → dict` (returns position/status/result), `async get_depth() → int`, background asyncio task `_process_loop()` that pulls from `asyncio.Queue`, acquires `asyncio.Lock`, calls Ollama via injected callable, uses `asyncio.wait_for(coro, timeout=self.timeout_seconds)` for per-request timeout. `InferenceRequest` dataclass with fields from data-model.md.
- [X] T012 [P] [US1] Implement `src/gpu_directer/server/api.py`: FastAPI app with `POST /gd/chat` (enqueues request, returns 202 with `{request_id, queue_position, status}`), `GET /gd/queue/{request_id}` (polls status/result), `GET /gd/health` (returns `{status, queue_depth, processing, ollama_reachable, gpu_available, uptime_seconds}`), `GET /gd/models` (proxies Ollama `/api/tags`). Startup event initializes `SerialQueue` from config. Import guards raise HTTP 503 if Ollama unreachable on startup.

### Server Doctor + Diagnostics

- [X] T013 [P] [US1] Implement `src/gpu_directer/server/doctor.py`: `run_doctor() → DiagnosticReport` performing 6 checks in order: (1) `docker_installed` — `subprocess.run(["docker", "--version"])`, (2) `ollama_container_running` — `subprocess.run(["docker", "inspect", "--format={{.State.Running}}", "ollama"])` → must be `"true"`, (3) `gpu_passthrough` — `subprocess.run(["docker", "exec", "ollama", "nvidia-smi"])` exit code 0, (4) `tailscale_connected` — `subprocess.run(["tailscale", "status", "--json"])` parse `BackendState == "Running"` and extract IPv4, (5) `ollama_models_available` — HTTP GET `http://localhost:11434/api/tags` returns ≥1 model, (6) `queue_status` — HTTP GET `http://localhost:11434/gd/health` returns queue depth. Each check returns `{name, status, detail, fix_hint}`. Fix hints match contracts/cli-schema.md examples.

### Server Setup Wizard

- [X] T014 [US1] Implement `src/gpu_directer/server/setup_wizard.py`: `run_server_setup(port=11434, non_interactive=False)` function executing 9 steps: (1) detect Docker via `doctor.check_docker()`, exit with fix hint if missing; (2) detect NVIDIA drivers via `subprocess.run(["nvidia-smi"])`; (3) pull `ollama/ollama` Docker image via `subprocess.run(["docker", "pull", "ollama/ollama"])`; (4) start container: `docker run -d --gpus all --name ollama -e OLLAMA_HOST=0.0.0.0:11434 -v ollama:/root/.ollama -p {port}:{port} --restart unless-stopped ollama/ollama`; (5) wait for Ollama ready (poll `/api/tags` up to 60s); (6) check/install Tailscale via `curl -fsSL https://tailscale.com/install.sh | sh`; (7) prompt user to run `sudo tailscale up` and press Enter when done; (8) get Tailscale IPv4 from `tailscale status --json`; (9) write `[server]` section to config, print IP and next steps. All steps print Rich-formatted status lines.

### Wire Up Server CLI Commands

- [X] T015 [US1] Wire `gpu-directer server setup` in `cli.py` to call `server.setup_wizard.run_server_setup(port, non_interactive)`. Display server setup wizard output with Rich formatting. Print Tailscale IP prominently on success.
- [X] T016 [US1] Wire `gpu-directer server doctor` in `cli.py` to call `server.doctor.run_doctor()` and display Rich-formatted table: `[✓]`/`[✗]` per check, detail text, fix hint indented below failing checks. `--json` flag outputs DiagnosticReport as JSON. Exit code 1 if any check fails.
- [X] T017 [US1] Wire `gpu-directer server models` in `cli.py` to HTTP GET `http://localhost:{port}/gd/models`, display model name/size/quantization in Rich table. `--json` flag returns raw JSON.
- [X] T018 [US1] Wire `gpu-directer server start`, `stop`, `restart` in `cli.py` to run `docker start|stop|restart ollama` via subprocess. Print success/failure with Rich formatting.

**Checkpoint**: On Ubuntu+NVIDIA: `gpu-directer server setup` completes without errors; `gpu-directer server doctor` shows all `[✓]`; `curl http://localhost:11434/api/tags` returns models

---

## Phase 4: US2 — Client Setup: Connect Client to Remote GPU Server (Priority: P1)

**Goal**: A user on any macOS/Ubuntu machine runs `gpu-directer client setup`, enters the server Tailscale IP, sees available models listed, and has a saved config. `gpu-directer client status` shows server online.

**Independent Test**: With a running server (Phase 3 complete): run `gpu-directer client setup --server-ip 100.x.x.x`, confirm models listed, then run `gpu-directer client status` — server shows `● online`.

- [X] T019 [P] [US2] Create `src/gpu_directer/client/connectivity.py`: `check_tailscale_installed() → bool` (FileNotFoundError on `tailscale --version`), `check_tailscale_connected() → dict` (parse `tailscale status --json`, return `{connected, own_ip}`), `probe_server(ip, port, timeout=5) → bool` (TCP `socket.connect_ex((ip, port)) == 0`), `query_server_models(ip, port) → list` (HTTP GET `/gd/models`, return model name list), `query_server_health(ip, port) → dict` (HTTP GET `/gd/health`, return health dict)
- [X] T020 [US2] Implement `src/gpu_directer/client/setup_wizard.py`: `run_client_setup(server_ip=None, port=11434, non_interactive=False)` executing: (1) check Tailscale installed, exit with fix hint if missing; (2) check Tailscale connected, prompt user to run `sudo tailscale up` if not; (3) prompt for server Tailscale IP if not provided; (4) TCP probe to `server_ip:port`, fail with diagnostic if unreachable; (5) query `/gd/models` and display available model list; (6) prompt for routing mode (default `auto`); (7) write `[client]` section to config.toml; (8) print summary with quickstart code example. Non-interactive mode uses provided flags or exits 1 with error.
- [X] T021 [US2] Implement `src/gpu_directer/client/status.py`: `get_client_status() → dict` querying remote server health (`/gd/health`), remote model list (`/gd/models`), and local Ollama health (`http://localhost:11434/api/tags`). Returns structured dict matching contracts/cli-schema.md `client status` output format.
- [X] T022 [US2] Wire `gpu-directer client setup` in `cli.py` to call `client.setup_wizard.run_client_setup()`. Display wizard steps with Rich formatting. Print quickstart code snippet on success.
- [X] T023 [US2] Wire `gpu-directer client status` in `cli.py` to call `client.status.get_client_status()` and display Rich-formatted output: server IP, `● online`/`● offline`, queue depth, routing mode, remote models, local Ollama status, local models. `--json` flag outputs raw dict. Exit code 1 if all sources offline.
- [X] T024 [US2] Wire `gpu-directer client models` in `cli.py` to call connectivity helpers for `--source remote|local|all` and display model table with source column. `--json` flag returns structured JSON.

**Checkpoint**: With running server: `gpu-directer client setup` completes; `gpu-directer client status` shows server online; config.toml has `[client]` section with server IP

---

## Phase 5: US3 — Unified API: GPURouter in Application Code (Priority: P1)

**Goal**: `from gpu_directer import GPURouter; router = GPURouter(); router.chat(model, messages)` routes to remote GPU server (auto mode), falls back to local Ollama if remote unreachable, emits warning if model missing on remote, raises `GPUDirecterConnectionError` if both unavailable.

**Independent Test**: Write a 5-line script using `GPURouter().chat()` with server online → routes to server. Kill server → same script routes to local Ollama (if available). Check that warning is emitted when model exists locally but not on remote.

- [X] T025 [P] [US3] Implement routing logic module `src/gpu_directer/client/routing.py`: `resolve_route(config, model, prefer=None) → str` returning `"remote"` or `"local"`. Implements the 4-step decision tree from contracts/python-api.md: (1) prefer/config mode check, (2) TCP probe remote, (3) model existence check via `/gd/models`, (4) local Ollama probe. Issues `warnings.warn()` for model-missing-on-remote case. Raises `GPUDirecterConnectionError` with descriptive message when no route available.
- [X] T026 [P] [US3] Implement `src/gpu_directer/client/poller.py`: `poll_for_result(base_url, request_id, timeout, poll_interval=1.0) → dict` that GETs `/gd/queue/{request_id}` every `poll_interval` seconds until `status` is `"complete"`, `"timeout"`, or `"error"`. Raises `GPUDirecterTimeoutError` on `status == "timeout"`. Returns Ollama-format ChatResponse dict on `"complete"`.
- [X] T027 [US3] Implement `src/gpu_directer/client/router.py`: `GPURouter` class. `__init__(config_path=None, routing_mode=None, timeout=None)` reads config via `config.load_config()`, sets `self.routing_mode`, `self.timeout`, `self.server_ip`, `self.server_port`. Raises `GPUDirecterConfigError` on missing/invalid config. `chat(model, messages, prefer=None, timeout=None, **kwargs) → ollama.ChatResponse`: calls `routing.resolve_route()`, if `"remote"` POSTs to `/gd/chat` then polls via `poller.poll_for_result()`, if `"local"` calls `ollama.Client(host="http://localhost:11434").chat(model, messages, **kwargs)`. `list_models(source="auto") → dict` calls connectivity helpers and returns `{remote: [...], local: [...], reachable: {remote, local}}`. `status() → dict` returns full status dict.
- [X] T028 [US3] Update `src/gpu_directer/__init__.py` to export `GPURouter` and all exception classes: `from gpu_directer.client.router import GPURouter`, `from gpu_directer.core.exceptions import *`
- [X] T029 [US3] Validate `GPURouter` drop-in compatibility: ensure `router.chat()` return type is `ollama.ChatResponse` (not a custom wrapper) so existing Ollama SDK usage is directly replaceable. Update `poller.py` to reconstruct `ollama.ChatResponse` from server response dict.

**Checkpoint**: `from gpu_directer import GPURouter; router = GPURouter(); r = router.chat("llama3.2", [{"role":"user","content":"hi"}]); print(r.message.content)` works end-to-end with remote server

---

## Phase 6: US4 — Serial Queue: Server-Side Concurrency Control (Priority: P2)

**Goal**: Multiple simultaneous inference requests from different clients all complete successfully; no GPU OOM. Queue depth visible; timeout releases slot.

**Independent Test**: Send 3 simultaneous requests to `POST /gd/chat`; verify all 3 return `status: "complete"` (not crash), server health shows correct queue depth during processing, 3rd request's queue position starts at 3 then decrements.

- [X] T030 [P] [US4] Connect `SerialQueue` config to `config.py`: update `server/api.py` startup to read `server.queue_timeout` and `server.max_queue_depth` from config.toml and pass to `SerialQueue.__init__()`. Add validation that `max_queue_depth=0` means unlimited.
- [X] T031 [US4] Implement queue full rejection in `server/api.py`: `POST /gd/chat` returns HTTP 503 with `{error: "Queue full", max_depth, current_depth}` when `SerialQueue.get_depth() >= max_depth` (and `max_depth > 0`). Wire `server.max_queue_depth` config setting.
- [X] T032 [US4] Add queue depth to `GET /gd/health` response and ensure `server/doctor.py` `queue_status` check reads live depth from running FastAPI process (via HTTP, not in-process).
- [X] T033 [US4] Verify `GPUDirecterTimeoutError` propagates correctly end-to-end: `server/queue.py` sets `status="timeout"` on `asyncio.TimeoutError`; `client/poller.py` raises `GPUDirecterTimeoutError` with message `"Request {request_id} exceeded queue timeout of {timeout}s"`.

**Checkpoint**: Concurrent requests queue correctly; queue depth visible in `gpu-directer client status`; timeout releases slot

---

## Phase 7: US5 — GitHub Install: pip install from GitHub (Priority: P2)

**Goal**: `pip install "git+https://github.com/.../GPU-Directer-toolkit.git[client]"` on a fresh macOS or Ubuntu machine with Python 3.8+ installs successfully with no manual steps. `gpu-directer --help` and `from gpu_directer import GPURouter` both work.

**Independent Test**: Create fresh Python 3.8 venv, run `pip install "git+https://...GPU-Directer-toolkit.git[client]"`, then run `gpu-directer --help` and `python -c "from gpu_directer import GPURouter"` — both succeed.

- [X] T034 [P] [US5] Finalize `pyproject.toml`: verify `[all]` extra correctly references both server and client, add `python_requires=">=3.8"`, add `long_description` from `README.md`, add `classifiers`, verify `console_scripts` entry point is correct. Confirm `tomli` conditional dependency (`; python_version < "3.11"`) works for 3.8–3.10.
- [X] T035 [P] [US5] Implement `--version` flag on `main` CLI group in `cli.py`: reads `__version__` from `gpu_directer.__version__`, prints `gpu-directer {version}`.
- [X] T036 [US5] Validate client-only install isolation: confirm `pip install gpu-directer[client]` does NOT install `fastapi` or `uvicorn`. Confirm `gpu-directer server setup` on a client-only install prints actionable error: `"Server dependencies not installed. Install with: pip install gpu-directer[server]"` and exits 1.
- [X] T037 [US5] Validate server-only install isolation: confirm `pip install gpu-directer[server]` does NOT break `import gpu_directer`. Confirm `from gpu_directer import GPURouter` on server-only install works (GPURouter only needs `ollama` which is a server dep).
- [X] T038 [US5] Add `.gitignore` covering `__pycache__/`, `*.egg-info/`, `dist/`, `build/`, `.venv/`, `~/.gpu-directer/` hint in README.

**Checkpoint**: Fresh venv install of `[client]`, `[server]`, and `[all]` extras all succeed; correct commands available per role

---

## Phase 8: US6 — Documentation: Follow Guide to Working System (Priority: P3)

**Goal**: A new user reads only the README and achieves a working end-to-end LLM call within 30 minutes.

**Independent Test**: User-test protocol: person unfamiliar with the project follows only README → working system in ≤30 minutes.

- [X] T039 [P] [US6] Write `README.md` Part 1: project overview, architecture diagram (ASCII), two-role explanation (gpu-server vs gpu-client), requirements table (server: Ubuntu+NVIDIA+Docker; client: macOS/Ubuntu+Python 3.8+)
- [X] T040 [P] [US6] Write `README.md` Part 2 — Tailscale Setup Guide (≤10 steps): create account, install on server (Ubuntu one-liner), install on client (macOS/Ubuntu), authenticate both devices, `tailscale ip -4` to get server IP, verify with ping
- [X] T041 [US6] Write `README.md` Part 3 — Server Quick-Start (≤15 steps): `pip install ...[server]`, `gpu-directer server setup`, `docker exec ollama ollama pull llama3.2`, `gpu-directer server doctor`
- [X] T042 [US6] Write `README.md` Part 4 — Client Quick-Start (≤15 steps): `pip install ...[client]`, `gpu-directer client setup`, `gpu-directer client status`, first `GPURouter` code example (3 lines)
- [X] T043 [US6] Write `README.md` Part 5 — Configuration Reference: complete `config.toml` schema table, `config set` examples for all common settings, explanation of routing modes (`auto`/`remote`/`local`)
- [X] T044 [US6] Write `README.md` Part 6 — Complete Command Reference: full command table matching `contracts/cli-schema.md`, Troubleshooting section covering top 4 failure scenarios (server unreachable, GPU not detected, queue full, model not found warning)

**Checkpoint**: README renders correctly on GitHub; all code examples are copy-pasteable and work; Tailscale guide is complete without external links needed

---

## Phase 9: Polish & Cross-Cutting Concerns

**Purpose**: Config CLI commands, global flags, output consistency across all commands

- [X] T045 [P] Implement `gpu-directer config show` in `cli.py`: call `config.load_config()`, format all sections as readable key=value pairs with Rich (or JSON with `--json`). Group by `[client]`, `[server]`, `[meta]` sections. Show config file path in header.
- [X] T046 [P] Implement `gpu-directer config set <key>=<value>` in `cli.py`: parse `key=value` argument, call `config.set_value(key, value)` which validates type/range, save config. Print confirmation `Set client.routing_mode = local`. Print error on unknown key or invalid value.
- [X] T047 [P] Implement `gpu-directer config edit` in `cli.py`: resolve `~/.gpu-directer/config.toml`, open with `os.environ.get("EDITOR", "nano")` via `subprocess.run()`. Create config file with defaults first if it doesn't exist.
- [X] T048 [P] Implement `gpu-directer config reset` in `cli.py`: prompt for confirmation (skip with `--yes`), overwrite config with `config.create_default_config()`. Print confirmation and path.
- [X] T049 Audit all CLI commands for consistent `--json` output: verify every command's `--json` path returns valid JSON to stdout with no extra text. Verify all error messages go to stderr regardless of `--json` flag.
- [X] T050 Audit all CLI commands for `--quiet` flag: suppress all Rich formatting and informational lines; only print final result or errors.
- [X] T051 Wire global `--config PATH` option in `cli.py` `main` group: use Click's `Context` to pass custom config path to all sub-commands. All `config.load_config()` calls respect this override.
- [X] T052 Run end-to-end validation against `quickstart.md`: follow every step in the quickstart guide, confirm each command produces exactly the output described, update quickstart if any steps diverge.

**Checkpoint**: All commands consistent; `--json` works everywhere; quickstart validated end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundation)**: Depends on Phase 1 — **BLOCKS all user story phases**
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 2; also requires a running server (Phase 3) for end-to-end testing only
- **Phase 5 (US3)**: Depends on Phase 2; requires Phase 3 server API (T011, T012) to exist
- **Phase 6 (US4)**: Depends on Phase 3 (server/queue.py must exist)
- **Phase 7 (US5)**: Depends on Phase 2 (pyproject.toml); can start after Phase 2
- **Phase 8 (US6)**: Depends on Phases 3–5 complete (documents working features)
- **Phase 9 (Polish)**: Depends on all story phases complete

### User Story Dependencies

- **US1 (P1)**: No story dependencies — first P1 to implement
- **US2 (P1)**: Independent of US1 for implementation; needs US1 for integration testing
- **US3 (P1)**: Depends on US1 server API (T011, T012 must exist); independent of US2
- **US4 (P2)**: Depends on US1 server queue (T011); adds hardening on top
- **US5 (P2)**: Depends only on Phase 2 foundation; can start after packaging skeleton done
- **US6 (P3)**: Depends on US1+US2+US3 complete (documents working features)

### Within Each Phase

- Tasks marked [P] can run simultaneously (different files, no blocking deps)
- Sequential tasks within a phase must complete in listed order
- Each phase produces an independently testable increment

### Parallel Opportunities

```
# Phase 1 — can run in parallel:
T003 (core/constants.py)    T004 (core/exceptions.py)

# Phase 2 — can run in parallel after T007:
T008 (server/__init__.py)    T009 (client/__init__.py)

# Phase 3 — can run in parallel after T011:
T012 (server/api.py)         T013 (server/doctor.py)

# Phase 4 — can run in parallel:
T019 (connectivity.py)

# Phase 5 — can run in parallel:
T025 (routing.py)            T026 (poller.py)

# Phase 6 — can run in parallel:
T030 (queue config)          T031 (queue full rejection)

# Phase 7 — can run in parallel:
T034 (pyproject.toml)        T035 (--version flag)

# Phase 8 — can run in parallel:
T039 (README overview)       T040 (Tailscale guide)

# Phase 9 — can run in parallel:
T045 (config show)  T046 (config set)  T047 (config edit)  T048 (config reset)
```

---

## Implementation Strategy

### MVP (US1 only — server ready to accept requests)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundation
3. Complete Phase 3: US1 (T011–T018)
4. **STOP and VALIDATE**: `gpu-directer server setup` + `server doctor` both work on Ubuntu+NVIDIA
5. Server is ready for client connections

### Incremental Delivery

1. **Phases 1–2** → Foundation ready
2. **Phase 3 (US1)** → Server functional; clients can connect manually with raw Ollama SDK
3. **Phase 4 (US2)** → Client setup wizard; `gpu-directer client status` works
4. **Phase 5 (US3)** → `GPURouter` API works; developers can use the toolkit in projects
5. **Phase 6 (US4)** → Concurrent requests safe; queue hardened
6. **Phase 7 (US5)** → GitHub install validated; open-source ready
7. **Phase 8 (US6)** → Full documentation; ready for community
8. **Phase 9 (Polish)** → Config CLI complete; production quality

### Solo Developer Strategy

Work sequentially in phase order. After each checkpoint, validate the increment works before proceeding. Phases 3, 4, and 5 each deliver a usable feature; any of them can be the "ship it" point depending on your needs.

---

## Notes

- [P] = different files with no dependency on incomplete tasks in same phase
- [USn] = maps to User Story n in spec.md
- No test tasks generated (TDD not requested in spec; add if desired)
- Each phase checkpoint is independently demonstrable
- Commit after each task or logical group to preserve history
- Config CLI (Phase 9) can be deferred if core workflow (Phases 1–5) is sufficient
