# GPU Access Router

[![Python](https://img.shields.io/badge/python-≥3.8-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Ubuntu-lightgrey)]()
[![Ollama](https://img.shields.io/badge/ollama-compatible-orange)](https://ollama.com)
[![Tailscale](https://img.shields.io/badge/network-Tailscale-blueviolet)](https://tailscale.com)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/huang422)

## What is GPU Access Router?

**GPU Access Router** is a lightweight Python toolkit that lets you run LLM inference on a remote GPU server from any machine — your laptop, a cloud VM, or anywhere else — with **zero code changes** to your existing Ollama workflow.

It solves a common problem: you have a powerful GPU machine at home or in the lab, but you develop on a laptop or cloud instance without a GPU. Instead of paying for cloud GPU or running slow CPU inference, GPU Access Router automatically routes your LLM calls to your own GPU over a secure Tailscale network, with transparent fallback to local Ollama when the server is unreachable.

### Key Features

- **Drop-in `ollama` module** — swap `import ollama` → `from gpu_access_router import ollama` and your code is environment-agnostic with zero other changes
- **`GPURouter` class** — finer-grained control over routing, fallback, and timeout
- **Auto-detection** — automatically finds and uses your remote GPU; falls back to local Ollama
- **`fallback_model`** — when remote fails, automatically retry with a smaller local model
- **Environment variable control** — switch environments without changing code (`GPU_ROUTER_ROUTING_MODE`, `GPU_ROUTER_SERVER_IP`, `GPU_ROUTER_FALLBACK_MODEL`)
- **Serial queue** — prevents GPU OOM by processing one inference request at a time
- **`ollama` CLI shim** — transparently proxies `ollama list`, `ollama run`, etc. to the remote GPU
- **TOML config** — simple per-environment configuration (great for conda/venv workflows)
- **Setup wizards** — interactive setup for both server and client
- **Health diagnostics** — 6-point `server doctor` check for quick troubleshooting

### How It Works

```
┌──────────────────────────────────────────────────────────┐
│                      Your Code (never changes)           │
│   from gpu_access_router import ollama                   │
│   response = ollama.chat("qwen3.5:35b-a3b", messages)    │
└───────────────────────┬──────────────────────────────────┘
                        │  routing decided by env vars / config
          ┌─────────────┴─────────────┐
          ▼                           ▼
 ┌─────────────────┐        ┌─────────────────────┐
 │  Remote GPU     │        │  Local Ollama       │
 │  Server         │        │  localhost:11434    │
 │  (Tailscale)    │        │                     │
 │                 │        │  fallback_model     │
 │  FastAPI :9090  │   ───► │  (if remote fails)  │
 │  Serial Queue   │        └─────────────────────┘
 │  + Ollama GPU   │
 └─────────────────┘
```

### Key Technical Design

<details>
<summary><strong>Serial Queue — GPU OOM Prevention</strong></summary>

Consumer GPUs (RTX 3090/4090, 24GB VRAM) crash when multiple inference requests run concurrently. The Serial Queue uses an `asyncio.Semaphore(1)` to ensure **exactly one request uses the GPU at a time**:

```
Client A ──┐
Client B ──┤──► SerialQueue (FIFO) ──► [Semaphore(1)] ──► Ollama GPU ──► Response
Client C ──┘
```

The server exposes two API styles that **share the same GPU semaphore**:

| Endpoint | Mode | Use Case |
|----------|------|----------|
| `/gd/chat` | Blocking | `GPURouter.chat()` — request enqueues, waits for `done_event`, returns full result |
| `/api/chat`, `/api/generate` | Streaming proxy | `AsyncClient` — acquires GPU slot, streams NDJSON via httpx, releases slot on completion |

Both paths are mutually exclusive — a streaming `/api/chat` blocks `/gd/chat` and vice versa.

</details>

<details>
<summary><strong>Model Hot-Swap & VRAM Management</strong></summary>

The queue tracks the currently loaded model (`_current_model`). When the next request uses a different model:

1. Calls `ollama.generate(model=old_model, prompt="", keep_alive=0)` to unload the old model from VRAM
2. Then proceeds with the new model inference

This prevents two models from occupying VRAM simultaneously.

</details>

<details>
<summary><strong>Sync (Streaming Proxy) Flow</strong></summary>

The `/api/*` endpoints act as a transparent Ollama-compatible proxy with queue protection:

1. Client sends standard Ollama request (e.g., `POST /api/chat`)
2. Server checks queue depth against `max_queue_depth`
3. `acquire_slot()` waits for the GPU semaphore
4. httpx streams the request to local Ollama (`:11434`)
5. Each NDJSON chunk is forwarded to the client in real-time
6. On stream completion (or error), `release_slot()` frees the semaphore

This means any Ollama-compatible client can connect to the server and get queue-protected GPU access without code changes — just point it to `http://<server-ip>:9090` instead of `localhost:11434`.

</details>

<details>
<summary><strong>4-Step Routing Decision Tree (Auto Mode)</strong></summary>

When `routing_mode=auto`, the client resolves the target using this logic:

1. **Probe remote** — TCP connect to `server_ip:server_port`
2. **Check remote models** — Query `/gd/models` for the requested model
3. **If model not on remote** — Check if local Ollama has it → route locally
4. **If remote unreachable** — Fall back to local Ollama silently

If `fallback_model` is configured, connection failures trigger an automatic retry with the fallback model on the local Ollama.

</details>

### Architecture: Two Roles

| Role             | Machine                        | What it does                                                                                       |
|------------------|--------------------------------|----------------------------------------------------------------------------------------------------|
| **gpu-server**   | Ubuntu + NVIDIA GPU            | Runs Ollama natively, exposes a queue-based FastAPI server on port `9090` over Tailscale |
| **gpu-client**   | macOS / Ubuntu (no GPU needed) | Connects to the server, routes `GPURouter.chat()` calls automatically                              |

| Port    | Service               | Scope                            |
|---------|-----------------------|----------------------------------|
| `11434` | Ollama native API     | Server-internal only             |
| `9090`  | GPU Access Router API | Exposed to clients via Tailscale |

---

## Quick Start

### Requirements

| Machine    | What you need                                            |
|------------|----------------------------------------------------------|
| GPU Server | Ubuntu 20.04+, NVIDIA GPU + drivers, Python 3.8+, Ollama |
| Client     | macOS 12+ or Ubuntu 20.04+, Python 3.8+                  |
| Both       | [Tailscale](https://tailscale.com) (free tier works)     |

### 1. Set Up Tailscale (both machines)

```bash
# GPU server (Ubuntu)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Client (macOS)
brew install tailscale    # or download from tailscale.com
sudo tailscale up

# Get the GPU server's Tailscale IP (run on the server)
tailscale ip -4
# Example output: 100.64.0.5

# Verify connectivity (from client)
ping 100.64.0.5
```

### 2. GPU Server Setup

```bash
# Install
pip install "gpu-access-router[server] @ git+https://github.com/huang422/GPU-Access-Router.git"
pip install --force-reinstall "gpu-access-router[server] @ git+https://github.com/huang422/GPU-Access-Router.git"

# Make sure Ollama is running with at least one model
ollama pull qwen3.5:9b

# Run the interactive setup wizard
gpu-access-router server setup

# Start the API server
gpu-access-router server serve
```

### 3. Client Setup

```bash
# Install
pip install "gpu-access-router[client] @ git+https://github.com/huang422/GPU-Access-Router.git"
pip install --force-reinstall "gpu-access-router[client] @ git+https://github.com/huang422/GPU-Access-Router.git"

# Connect to your GPU server (use port 9090, not 11434)
gpu-access-router client setup
gpu-access-router client setup --server-ip 100.64.0.5 --port 9090

# Verify
gpu-access-router client status
```

### 4. Use It

```python
# Drop-in replacement for `import ollama` — works on any environment
from gpu_access_router import ollama

response = ollama.chat(
    model="qwen3.5:35b-a3b",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print(response.message.content)
```

Routing is controlled by environment variables — no code changes when switching environments:

```bash
# Local Ollama only
export GPU_ROUTER_ROUTING_MODE=local

# Remote GPU, auto-fallback to small local model if remote fails
export GPU_ROUTER_SERVER_IP=100.64.0.5
export GPU_ROUTER_ROUTING_MODE=auto
export GPU_ROUTER_FALLBACK_MODEL=qwen3.5:9b
```

---

## Python API

### Drop-in `ollama` module — Sync (recommended for scripts)

```python
# Replace: import ollama
# With:
from gpu_access_router import ollama

# API is identical to the native ollama module
response = ollama.chat("qwen3.5:35b-a3b", [{"role": "user", "content": "Hello!"}])
print(response.message.content)

response = ollama.generate("qwen3.5:35b-a3b", "Tell me a joke")
print(response.response)

models = ollama.list()

# ollama.Client drop-in — accepts routing_mode and fallback_model
client = ollama.Client(routing_mode="auto", fallback_model="qwen3.5:9b")
response = client.chat("qwen3.5:35b-a3b", [...])
```

### AsyncClient — Async (recommended for web apps & bots)

Drop-in replacement for `ollama.AsyncClient` with automatic routing and fallback.
Install: `pip install "gpu-access-router[client]"`

```python
# Replace: from ollama import AsyncClient
# With:
from gpu_access_router.ollama import AsyncClient

async with AsyncClient() as client:
    # Streaming
    response = await client.generate("qwen3.5:9b", "Hello", stream=True)
    async for chunk in response:
        print(chunk.response, end="")

    # Non-streaming
    resp = await client.generate("qwen3.5:9b", "Hello")
    print(resp.response)

    # Think mode (reasoning models)
    resp = await client.generate("qwen3.5:35b-a3b", "Analyze ML trends", think=True)
    print(resp.thinking)  # reasoning content
    print(resp.response)  # final answer

    # Multimodal (vision models)
    resp = await client.generate(
        "qwen3.5:35b-a3b", "Describe this image",
        images=[base64_data], think=False,
    )

    # All Ollama parameters pass through: system, options, format, etc.
    resp = await client.generate(
        "qwen3.5:35b-a3b", "Explain quantum computing",
        system="You are a physics professor.",
        options={"num_predict": 6144, "temperature": 0.7, "num_ctx": 8192},
    )

    # List models
    models = await client.list()
```

**Constructor options:**

```python
client = AsyncClient(
    routing_mode="auto",          # "auto" | "remote" | "local" (default: from config)
    fallback_model="qwen3.5:9b",  # Local model when remote fails (default: from config)
    timeout=300,                  # Queue timeout in seconds (default: from config)
)
```

**Module-level async functions:**

```python
from gpu_access_router import ollama

# One-liner async calls (use a shared singleton client)
resp = await ollama.agenerate("qwen3.5:9b", "Hello", stream=True)
resp = await ollama.achat("qwen3.5:35b-a3b", [{"role": "user", "content": "Hi"}])
models = await ollama.alist()
```

**How routing works:**

| Scenario | Behaviour |
|----------|-----------|
| Remote GPU reachable + model available | Routes to remote GPU |
| Remote GPU reachable + model NOT available but local has it | Routes to local Ollama |
| Remote GPU unreachable + local Ollama running | Falls back to local Ollama |
| Remote fails mid-request + `fallback_model` set | Retries with fallback model on local |
| Nothing available | Raises `GPUAccessRouterConnectionError` |

### GPURouter (lower-level control)

```python
from gpu_access_router import GPURouter

router = GPURouter()

# Auto-route: remote GPU first, local Ollama fallback
response = router.chat(model="qwen3.5:35b-a3b", messages=[...])

# With fallback model — if remote fails, use this local model instead
response = router.chat(model="qwen3.5:35b-a3b", messages=[...], fallback_model="qwen3.5:9b")

# Force remote only (raises GPUAccessRouterConnectionError if unreachable)
response = router.chat(model="qwen3.5:35b-a3b", messages=[...], prefer="remote")

# Force local only
response = router.chat(model="qwen3.5:35b-a3b", messages=[...], prefer="local")

# List models on remote and local
models = router.list_models()
print("Remote:", [m["name"] for m in models["remote"] or []])
print("Local:",  [m["name"] for m in models["local"] or []])

# Check status
status = router.status()
print("Queue depth:", status["remote"].get("queue_depth", 0))
```

### Exception Handling

```python
from gpu_access_router import (
    GPURouter,
    GPUAccessRouterConnectionError,
    GPUAccessRouterTimeoutError,
)

router = GPURouter()
try:
    response = router.chat(model="qwen3.5:35b-a3b", messages=[...], prefer="remote")
except GPUAccessRouterConnectionError:
    print("Server unreachable")
except GPUAccessRouterTimeoutError:
    print("Request timed out in queue")
```

---

## ollama CLI Shim

When installed in a conda/venv environment, GPU Access Router provides a transparent `ollama` shim that routes commands to the remote GPU server.

```bash
conda activate my-gpu-env

ollama list                            # Lists models on REMOTE server
ollama run qwen3.5:9b "Hello world"    # Runs inference on REMOTE GPU
ollama ps                              # Shows remote queue status
ollama show qwen3.5:9b                 # Shows remote model details

ollama pull qwen3.5:9b                 # Prints SSH instructions
ollama run qwen3.5:9b                  # No prompt = interactive, falls through to local
```

| Command                                | Behaviour                                        |
|----------------------------------------|--------------------------------------------------|
| `ollama list` / `ollama ls`            | List models on the **remote** GPU server         |
| `ollama ps`                            | Show remote queue depth and processing status    |
| `ollama show <model>`                  | Show remote model details                        |
| `ollama run <model> "prompt"`          | Run inference on the **remote** GPU              |
| `ollama pull/push/rm/stop/cp/create`   | Print SSH instructions to run on server          |
| `ollama run <model>` (no prompt)       | Fall through to local ollama (interactive)       |
| Unknown commands                       | Fall through to local ollama                     |

---

## CLI Reference

### Server commands

```bash
gpu-access-router server setup                    # Interactive setup wizard
gpu-access-router server serve [--host] [--port]  # Start API server (foreground)
gpu-access-router server start                    # Start in background
gpu-access-router server stop                     # Stop background server
gpu-access-router server restart                  # Restart background server
gpu-access-router server doctor [--json]          # 6-point health check
gpu-access-router server models [--json]          # List models
```

### Client commands

```bash
gpu-access-router client setup [--server-ip IP] [--port PORT] [--fallback-model MODEL]   # Setup wizard
gpu-access-router client status [--json]                        # Connection status
gpu-access-router client models [--source remote|local|all]     # List models
```

### Config commands

```bash
gpu-access-router config show [--json]            # Print config
gpu-access-router config set <key>=<value>        # Update a value
gpu-access-router config edit                     # Open in $EDITOR
gpu-access-router config reset [--yes]            # Reset to defaults
```

### Global flags

| Flag            | Description              |
|-----------------|--------------------------|
| `--version`     | Print version and exit   |
| `--json`        | JSON output              |
| `--quiet`       | Suppress info messages   |
| `--config PATH` | Custom config file path  |

---

## Configuration

Config file: `~/.gpu-access-router/config.toml`

```toml
[client]
server_ip       = "100.64.0.5"    # Tailscale IP of GPU server
server_port     = 9090            # GPU Access Router API port
routing_mode    = "auto"          # "auto" | "remote" | "local"
timeout_seconds = 300             # Queue wait timeout
default_model   = ""              # Optional default model
fallback_model  = "qwen3.5:9b"    # Local model to use when remote fails (optional)

[server]
ollama_port     = 11434           # Internal Ollama port
api_port        = 9090            # FastAPI server port
queue_timeout   = 300             # Max queue wait time
max_queue_depth = 10              # Max queued requests (0 = unlimited)

[meta]
role            = "client"        # "client" | "server" | "both"
version         = "0.1.0"
```

### Environment variables (override config without editing the file)

| Variable | Description |
|----------|-------------|
| `GPU_ROUTER_SERVER_IP` | Remote GPU server Tailscale IP |
| `GPU_ROUTER_ROUTING_MODE` | `auto` \| `local` \| `remote` |
| `GPU_ROUTER_FALLBACK_MODEL` | Local model used when remote fails |
| `GPU_ACCESS_ROUTER_CONFIG` | Path to a custom config file |

```bash
# Local laptop — only local Ollama
export GPU_ROUTER_ROUTING_MODE=local

# Remote GPU — prefer remote, fallback to small model if it fails
export GPU_ROUTER_SERVER_IP=100.64.0.5
export GPU_ROUTER_ROUTING_MODE=auto
export GPU_ROUTER_FALLBACK_MODEL=qwen3.5:9b

# Force remote, no fallback
export GPU_ROUTER_SERVER_IP=100.64.0.5
export GPU_ROUTER_ROUTING_MODE=remote
```

### Per-environment config (conda / venv)

Use `GPU_ACCESS_ROUTER_CONFIG` to give each environment its own config:

```bash
# Create env-specific config
conda activate rgpu
gpu-access-router --config ~/.gpu-access-router/rgpu.toml config set client.server_ip=100.64.0.5
gpu-access-router --config ~/.gpu-access-router/rgpu.toml config set client.routing_mode=remote

# Auto-load on conda activate
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d

echo 'export GPU_ACCESS_ROUTER_CONFIG="$HOME/.gpu-access-router/rgpu.toml"' \
  > $CONDA_PREFIX/etc/conda/activate.d/gpu_access_router.sh

echo 'unset GPU_ACCESS_ROUTER_CONFIG' \
  > $CONDA_PREFIX/etc/conda/deactivate.d/gpu_access_router.sh
```

| Environment            | Config        | ollama command        | Routing |
|------------------------|---------------|-----------------------|---------|
| `conda activate rgpu`  | `rgpu.toml`   | Proxied to remote GPU | Remote  |
| Other environments     | `config.toml` | Real system ollama    | Local   |

---

## Developer Setup

```bash
git clone https://github.com/huang422/GPU-Access-Router.git
cd GPU-Access-Router
pip install -e ".[all]"
gpu-access-router --version
```

### Update GPU server after pushing changes

```bash
# On the GPU server:
pip install --force-reinstall "gpu-access-router[server] @ git+https://github.com/huang422/GPU-Access-Router.git"
gpu-access-router server restart
```

---

## Troubleshooting

### Server unreachable

```bash
tailscale status                    # Check Tailscale on both machines
gpu-access-router server doctor     # Run on server
gpu-access-router client status     # Run on client
curl http://<tailscale-ip>:9090/gd/health
```

### ollama shim not working

```bash
which ollama                        # Should show env path, not /usr/local/bin/ollama
echo $GPU_ACCESS_ROUTER_CONFIG      # Should be set in the right env
conda deactivate && conda activate rgpu
```

### Queue full or timeouts

```bash
gpu-access-router client status
gpu-access-router config set client.timeout_seconds=600
gpu-access-router config set server.queue_timeout=600
```

### Model not found

```bash
# Pull on the server (SSH in first):
ollama pull qwen3.5:9b
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or collaboration inquiries:

- Developer: Tom Huang
- Email: huang1473690@gmail.com
