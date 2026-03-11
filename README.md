# GPU Access Router

[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Ubuntu-lightgrey)]()
[![Ollama](https://img.shields.io/badge/ollama-compatible-orange)](https://ollama.com)
[![Tailscale](https://img.shields.io/badge/network-Tailscale-blueviolet)](https://tailscale.com)

## What is GPU Access Router?

**GPU Access Router** is a lightweight Python toolkit that lets you run LLM inference on a remote GPU server from any machine — your laptop, a cloud VM, or anywhere else — with **zero code changes** to your existing Ollama workflow.

It solves a common problem: you have a powerful GPU machine at home or in the lab, but you develop on a laptop or cloud instance without a GPU. Instead of paying for cloud GPU or running slow CPU inference, GPU Access Router automatically routes your LLM calls to your own GPU over a secure Tailscale network, with transparent fallback to local Ollama when the server is unreachable.

### Key Features

- **Single Python class** — `GPURouter` replaces direct Ollama calls with smart routing
- **Auto-detection** — automatically finds and uses your remote GPU; falls back to local Ollama
- **Serial queue** — prevents GPU OOM by processing one inference request at a time
- **`ollama` CLI shim** — transparently proxies `ollama list`, `ollama run`, etc. to the remote GPU
- **TOML config** — simple per-environment configuration (great for conda/venv workflows)
- **Setup wizards** — interactive setup for both server and client
- **Health diagnostics** — 6-point `server doctor` check for quick troubleshooting

### How It Works

```
┌──────────────────────────────────────────────────────────┐
│                      Your Code                           │
│   from gpu_access_router import GPURouter                │
│   router = GPURouter()                                   │
│   response = router.chat("qwen3.5:9b", messages)           │
└───────────────────────┬──────────────────────────────────┘
                        │  auto-route
          ┌─────────────┴─────────────┐
          ▼                           ▼
 ┌─────────────────┐        ┌─────────────────────┐
 │  Remote GPU     │        │  Local Ollama       │
 │  Server         │        │  (fallback)         │
 │  (Tailscale)    │        │  localhost:11434    │
 │                 │        └─────────────────────┘
 │  FastAPI :8080  │
 │  Serial Queue   │
 │  + Ollama GPU   │
 └─────────────────┘
```

### Architecture: Two Roles

| Role             | Machine                        | What it does                                                                                       |
|------------------|--------------------------------|----------------------------------------------------------------------------------------------------|
| **gpu-server**   | Ubuntu + NVIDIA GPU            | Runs Ollama (native or Docker), exposes a queue-based FastAPI server on port `8080` over Tailscale |
| **gpu-client**   | macOS / Ubuntu (no GPU needed) | Connects to the server, routes `GPURouter.chat()` calls automatically                              |

| Port    | Service               | Scope                            |
|---------|-----------------------|----------------------------------|
| `11434` | Ollama native API     | Server-internal only             |
| `8080`  | GPU Access Router API | Exposed to clients via Tailscale |

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

# Connect to your GPU server (use port 8080, not 11434)
gpu-access-router client setup
gpu-access-router client setup --server-ip 100.64.0.5 --port 8080

# Verify
gpu-access-router client status
```

### 4. Use It

```python
from gpu_access_router import GPURouter

router = GPURouter()
response = router.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)
print(response.message.content)
```

That's it. The router automatically sends inference to your remote GPU. If the server is unreachable, it falls back to local Ollama.

---

## Python API

### GPURouter

```python
from gpu_access_router import GPURouter

router = GPURouter()

# Auto-route (default): remote GPU first, local Ollama fallback
response = router.chat(model="qwen3.5:9b", messages=[...])

# Force remote only (raises GPUAccessRouterConnectionError if unreachable)
response = router.chat(model="qwen3.5:9b", messages=[...], prefer="remote")

# Force local only
response = router.chat(model="qwen3.5:9b", messages=[...], prefer="local")

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
    response = router.chat(model="qwen3.5:9b", messages=[...], prefer="remote")
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

ollama list                          # Lists models on REMOTE server
ollama run qwen3.5:9b "Hello world"    # Runs inference on REMOTE GPU
ollama ps                            # Shows remote queue status
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
gpu-access-router client setup [--server-ip IP] [--port PORT]   # Setup wizard
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
server_ip       = "100.64.0.5"   # Tailscale IP of GPU server
server_port     = 8080            # GPU Access Router API port
routing_mode    = "auto"          # "auto" | "remote" | "local"
timeout_seconds = 300             # Queue wait timeout
default_model   = ""              # Optional default model

[server]
ollama_port     = 11434           # Internal Ollama port
api_port        = 8080            # FastAPI server port
queue_timeout   = 300             # Max queue wait time
max_queue_depth = 10              # Max queued requests (0 = unlimited)

[meta]
role            = "client"        # "client" | "server" | "both"
version         = "0.1.0"
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
curl http://<tailscale-ip>:8080/gd/health
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
# Or via Docker:
docker exec ollama ollama pull qwen3.5:9b
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or collaboration inquiries:

- Developer: Tom Huang
- Email: huang1473690@gmail.com
