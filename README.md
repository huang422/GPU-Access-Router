# GPU Directer

[![Python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20|%20Ubuntu-lightgrey)]()
[![Ollama](https://img.shields.io/badge/ollama-compatible-orange)](https://ollama.com)
[![Tailscale](https://img.shields.io/badge/network-Tailscale-blueviolet)](https://tailscale.com)

Route LLM inference calls to a remote GPU server or fall back to local Ollama automatically — with a single Python class, a simple CLI, and a transparent `ollama` command shim.

```
┌──────────────────────────────────────────────────────────┐
│                      Your Code                           │
│   from gpu_directer import GPURouter                     │
│   router = GPURouter()                                   │
│   response = router.chat("llama3.2", messages)           │
└───────────────────────┬──────────────────────────────────┘
                        │  auto-route
          ┌─────────────┴─────────────┐
          ▼                           ▼
 ┌─────────────────┐        ┌─────────────────────┐
 │  Remote GPU     │        │  Local Ollama        │
 │  Server         │        │  (fallback)          │
 │  (Tailscale)    │        │  localhost:11434      │
 │                 │        └─────────────────────┘
 │  FastAPI :8080  │
 │  Serial Queue   │
 │  + Ollama GPU   │
 └─────────────────┘
```

## Two Roles

| Role | Machine | Purpose |
|---|---|---|
| **gpu-server** | Ubuntu + NVIDIA GPU | Runs Ollama (native or Docker), exposes queue API on port 8080 over Tailscale |
| **gpu-client** | macOS or Ubuntu (no GPU needed) | Connects to server port 8080, routes `GPURouter.chat()` calls |

## Port Architecture

| Port | Service | Where |
|---|---|---|
| `11434` | Ollama | Server only (internal, not exposed to clients) |
| `8080` | GPU Directer FastAPI server | Server → exposed to clients via Tailscale |

## Requirements

| Machine | Requirements |
|---|---|
| GPU Server | Ubuntu 20.04+, NVIDIA GPU + drivers, Python 3.8+, Ollama (native or Docker) |
| Client | macOS 12+ or Ubuntu 20.04+, Python 3.8+ |

---

## Part 1: Tailscale Setup (both machines)

Tailscale creates a secure private network between your machines.

1. Sign up at [tailscale.com](https://tailscale.com) (free tier is sufficient)

2. **On the GPU server (Ubuntu)**:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```
   Follow the printed URL to authenticate.

3. **On the client (macOS)**:
   Download from [tailscale.com/download](https://tailscale.com/download) or:
   ```bash
   brew install tailscale
   sudo tailscale up
   ```
   **Ubuntu client**: same as the server.

4. **Get the GPU server's Tailscale IP**:
   ```bash
   # On the GPU server:
   tailscale ip -4
   # Outputs something like: 100.64.0.5
   ```
   Save this IP — you'll need it during client setup.

5. **Verify connectivity** (from client):
   ```bash
   ping -c 3 100.64.0.5   # use your actual Tailscale IP
   ```

---

## Part 2: GPU Server Setup

### Install

```bash
pip install "gpu-directer[server] @ git+https://github.com/huang422/GPU-Directer.git"
```

### Ollama

GPU Directer connects to Ollama on `localhost:11434`. If Ollama is already running (native install), no Docker setup is needed.

```bash
# If Ollama is not installed yet:
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3.2

# Verify Ollama is up
curl http://localhost:11434/api/tags
```

### Run the setup wizard

```bash
gpu-directer server setup
```

The wizard:
- Checks Docker and NVIDIA drivers
- **Detects if Ollama is already running** — skips Docker setup if so
- If Ollama is not running, pulls and starts `ollama/ollama` Docker container with `--gpus all`
- Installs Tailscale if needed and prompts you to authenticate
- Prints your Tailscale IP and writes `~/.gpu-directer/config.toml`

### Start the GPU Directer API server

After setup, start the FastAPI queue server (this is what clients connect to):

```bash
# Start in foreground
gpu-directer server serve

# Or specify host/port explicitly
gpu-directer server serve --host 0.0.0.0 --port 8080
```

To keep it running in the background, use `nohup` or a systemd service:

```bash
nohup gpu-directer server serve > ~/gpu-directer.log 2>&1 &
```

### Verify everything is healthy

```bash
gpu-directer server doctor
```

Expected output:
```
[✓] Docker installed          Docker 24.0.7
[✓] Ollama container running  Container 'ollama' is running
[✓] GPU passthrough active    NVIDIA GeForce RTX 4090
[✓] Tailscale connected       Connected. Tailscale IP: 100.64.0.5
[✓] Models available          1 model: llama3.2
[✓] Queue status              Queue depth: 0, Processing: idle

Overall: PASS
```

> **Note**: "Ollama container running" check uses Docker. If you run Ollama natively, this check will show fail but "Queue status" and "Models available" will still pass.

---

## Part 3: Client Setup

```bash
# Install (on your laptop / cloud machine)
pip install "gpu-directer[client] @ git+https://github.com/huang422/GPU-Directer.git"

# Run the interactive setup wizard
# Use --port 8080 (GPU Directer API port, not Ollama's 11434)
gpu-directer client setup --server-ip 100.64.0.5 --port 8080

# Verify the connection
gpu-directer client status
```

Expected output:
```
GPU Directer Client Status
  Server IP:      100.64.0.5:8080
  Server status:  ● online
  Queue depth:    0 requests waiting
  Routing mode:   auto
  Available models (remote):
    llama3.2
  Local Ollama:   ● offline
```

---

## Part 4: Test the Connection (Client)

After client setup, verify everything works end-to-end.

### Step 1 — Check connectivity

```bash
gpu-directer client status       # server reachable + queue depth
gpu-directer client models       # list remote and local models
```

### Step 2 — Run inference from Python

```python
from gpu_directer import GPURouter

router = GPURouter()

# Quick status check
status = router.status()
print("Remote reachable:", status["remote"]["reachable"])
print("Models:", [m["name"] for m in status["remote"].get("models", [])])

# Send a request — auto-routes to remote GPU server
response = router.chat(
    model="qwen3.5:9b",   # replace with any model pulled on the server
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
print(response.message.content)
```

### Step 3 — Full end-to-end test script

Save as `test_gpu.py` and run from your client machine:

```python
from gpu_directer import GPURouter, GPUDirecterConnectionError

router = GPURouter()

# Print status
s = router.status()
print(f"Server:    {s['remote']['server_ip']}:{s['remote']['port']}")
print(f"Reachable: {s['remote']['reachable']}")
print(f"Queue:     {s['remote'].get('queue_depth', 0)} waiting")
print(f"Models:    {[m['name'] for m in s['remote'].get('models', [])]}")
print()

# Force remote inference
try:
    response = router.chat(
        model="qwen3.5:9b",
        messages=[{"role": "user", "content": "Reply with just: OK"}],
        prefer="remote",
    )
    print("✓ Remote inference success:", response.message.content.strip())
except GPUDirecterConnectionError as e:
    print(f"✗ Remote not available: {e}")
```

```bash
python test_gpu.py
```

---

## Part 5: Use GPURouter in Your Code

```python
from gpu_directer import GPURouter

router = GPURouter()

response = router.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}]
)
print(response.message.content)
```

### Routing modes

```python
msgs = [{"role": "user", "content": "Hello!"}]

# Auto (default): try remote first, fall back to local
response = router.chat(model="qwen3.5:9b", messages=msgs)

# Force remote only (raises GPUDirecterConnectionError if unreachable)
response = router.chat(model="qwen3.5:9b", messages=msgs, prefer="remote")

# Force local only
response = router.chat(model="qwen3.5:9b", messages=msgs, prefer="local")
```

### Check what's available

```python
models = router.list_models()
print("Remote:", [m["name"] for m in models["remote"] or []])
print("Local:",  [m["name"] for m in models["local"] or []])

status = router.status()
print("Queue depth:", status["remote"].get("queue_depth", 0))
print("Routing mode:", status["routing_mode"])
```

---

## Part 6: ollama CLI Shim

When gpu-directer is installed in a conda/venv environment, it installs a transparent `ollama` shim that **replaces the system `ollama` command inside that environment**. All `ollama` commands automatically target the remote GPU server.

### Command routing

| Command | Behaviour |
|---|---|
| `ollama list` / `ollama ls` | List models on the **remote** GPU server |
| `ollama ps` | Show remote queue depth and processing status |
| `ollama show <model>` | Show remote model details (size, digest, family) |
| `ollama run <model> "prompt"` | Run single-shot inference on **remote** GPU |
| `ollama pull <model>` | Print SSH instructions to pull on the server |
| `ollama push <model>` | Print SSH instructions to push from the server |
| `ollama rm <model>` | Print SSH instructions to remove on the server |
| `ollama stop <model>` | Print SSH instructions to stop on the server |
| `ollama cp <src> <dst>` | Print SSH instructions to copy on the server |
| `ollama create <name>` | Print SSH instructions to create on the server |
| `ollama serve` | Print SSH note (server is already running) |
| `ollama run <model>` (no prompt) | Pass through to local ollama (interactive mode) |
| Any other command | Pass through to local ollama |

### Example

```bash
conda activate rgpu

ollama list
# NAME                                            ID              SIZE      MODIFIED
# qwen3.5:9b                                      a1b2c3d4e5f6    5.4 GB    2026-03-10 14:22:00

ollama ps
# Remote GPU server  100.64.0.5:8080
#   Status:    idle
#   Queue:     0 waiting
#   Uptime:    2h 15m

ollama run qwen3.5:9b "用繁體中文解釋什麼是機器學習"
# 機器學習是人工智能的一個分支...

ollama pull llama3.2
# [gpu-directer] 'pull' must run on the remote GPU server.
#   SSH in and run:
#     ssh <user>@100.64.0.5
#     ollama pull llama3.2
```

---

## Part 7: Configuration Reference

Config is stored at `~/.gpu-directer/config.toml`:

```toml
[client]
server_ip       = "100.64.0.5"   # Tailscale IP of the GPU server
server_port     = 8080            # GPU Directer API port (default: 8080)
routing_mode    = "auto"          # "auto" | "remote" | "local"
timeout_seconds = 300             # Queue wait timeout in seconds
default_model   = ""              # Optional: pre-select a model

[server]
ollama_port     = 11434           # Port Ollama listens on (internal only)
api_port        = 8080            # Port GPU Directer FastAPI server listens on
queue_timeout   = 300             # Max seconds a request waits in queue
max_queue_depth = 10              # Max requests in queue (0 = unlimited)

[meta]
role            = "client"        # "client" | "server" | "both"
version         = "0.1.0"
created_at      = "2026-03-11T00:00:00"
```

### Common config commands

```bash
# View all settings
gpu-directer config show

# Update a value
gpu-directer config set client.routing_mode=local
gpu-directer config set client.server_ip=100.64.0.99
gpu-directer config set client.timeout_seconds=600
gpu-directer config set server.queue_timeout=120
gpu-directer config set server.api_port=8080

# Open in $EDITOR
gpu-directer config edit

# Reset to defaults
gpu-directer config reset
```

### Per-environment config (conda / venv)

By default all environments share `~/.gpu-directer/config.toml`.
Use `GPU_DIRECTER_CONFIG` to give each conda/venv environment its **own** config — so only the `rgpu` environment connects to the remote GPU server.

**Step 1 — Create the rgpu config**

```bash
conda activate rgpu

# Create a separate config for this environment
gpu-directer --config ~/.gpu-directer/rgpu.toml config set client.server_ip=100.64.0.5
gpu-directer --config ~/.gpu-directer/rgpu.toml config set client.routing_mode=remote
```

**Step 2 — Set default config to local-only**

```bash
# In your base / default environment
gpu-directer config set client.routing_mode=local
```

**Step 3 — Auto-load config when activating the conda env**

```bash
conda activate rgpu

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

echo 'export GPU_DIRECTER_CONFIG="$HOME/.gpu-directer/rgpu.toml"' \
  > $CONDA_PREFIX/etc/conda/activate.d/gpu_directer.sh

echo 'unset GPU_DIRECTER_CONFIG' \
  > $CONDA_PREFIX/etc/conda/deactivate.d/gpu_directer.sh
```

**Result**

| Environment | Config loaded | ollama command | Routing |
|---|---|---|---|
| `conda activate rgpu` | `~/.gpu-directer/rgpu.toml` | proxies to remote GPU | Remote GPU server |
| base / other envs | `~/.gpu-directer/config.toml` | real system ollama | Local Ollama |
| `conda deactivate` | env var cleared | real system ollama | Local Ollama |

> **Note**: The same approach works with venv — place the `export` in `<venv>/bin/activate` and `unset` in `<venv>/bin/deactivate`.
>
> `GPU_DIRECTER_CONFIG` is also respected by `GPURouter()` in Python — no code changes needed.

---

## Part 8: Complete Command Reference

### Server commands (requires `[server]` install)

| Command | Description |
|---|---|
| `gpu-directer server setup [--port PORT] [--api-port PORT]` | Run setup wizard (auto-starts API server at the end) |
| `gpu-directer server start` | Start GPU Directer API server in background |
| `gpu-directer server stop` | Stop GPU Directer API server |
| `gpu-directer server restart` | Restart GPU Directer API server |
| `gpu-directer server serve [--host HOST] [--port PORT] [--reload]` | Start API server in foreground (dev/debug mode) |
| `gpu-directer server doctor [--json]` | 6-point health check |
| `gpu-directer server models [--json]` | List models (requires API server running) |

### Client commands (requires `[client]` install)

| Command | Description |
|---|---|
| `gpu-directer client setup [--server-ip IP] [--port PORT] [--non-interactive]` | Interactive client setup |
| `gpu-directer client status [--json]` | Show server + local Ollama status |
| `gpu-directer client models [--source remote\|local\|all] [--json]` | List available models |

### Config commands (all installs)

| Command | Description |
|---|---|
| `gpu-directer config show [--json]` | Print current config |
| `gpu-directer config set <key>=<value>` | Update one setting |
| `gpu-directer config edit` | Open config in `$EDITOR` |
| `gpu-directer config reset [--yes]` | Reset to defaults |

### Global flags

| Flag | Description |
|---|---|
| `--version` | Print version and exit |
| `--json` | Machine-readable JSON output |
| `--quiet` | Suppress informational output |
| `--config PATH` | Use a custom config file |

---

## Part 9: Developer Workflow

### Initial setup (dev machine, editable install)

```bash
git clone https://github.com/huang422/GPU-Directer.git
cd GPU-Directer
pip install -e ".[all]"          # installs both server + client extras
gpu-directer --version           # verify install
```

### Update after code changes

**Dev machine** (editable install — Python changes take effect immediately, no reinstall needed):
```bash
git pull                         # pull latest changes
# No reinstall needed for .py changes
# Only reinstall if pyproject.toml dependencies changed:
pip install -e ".[all]"
```

**GPU server** (re-install from GitHub after pushing):
```bash
# First push your changes from dev machine:
git add -A && git commit -m "your message" && git push

# Then on the GPU server:
pip install --force-reinstall "gpu-directer[server] @ git+https://github.com/huang422/GPU-Directer.git"

# Restart the API server after update:
pkill -f "gpu-directer server serve"
gpu-directer server serve &
```

Alternatively, clone on the server for faster iteration:
```bash
# On GPU server (one-time setup):
git clone https://github.com/huang422/GPU-Directer.git
cd GPU-Directer
pip install -e ".[server]"

# Update:
git pull                         # changes take effect immediately
pkill -f "gpu-directer server serve"
gpu-directer server serve &
```

### Verify the full stack locally

```bash
# 1. Check imports
python -c "from gpu_directer import GPURouter; print('OK')"

# 2. Check config
gpu-directer config show

# 3. Check client can reach server
gpu-directer client status

# 4. Run linter
cd src && ruff check .
```

---

## Troubleshooting

### Server unreachable from client

```bash
tailscale status                  # Check Tailscale is running on both machines
gpu-directer server doctor        # Run on server — check all 6 items
gpu-directer client status        # Run on client — shows connectivity

# Make sure the API server is running on the GPU server:
gpu-directer server serve         # Should be running in background
curl http://<tailscale-ip>:8080/gd/health
```

### ollama shim not taking effect

```bash
# Check which ollama is active
which ollama
# Should show: .../envs/rgpu/bin/ollama
# If it shows /usr/local/bin/ollama, reinstall:
pip install --force-reinstall --no-cache-dir \
  "gpu-directer[client] @ git+https://github.com/huang422/GPU-Directer.git"

# Check GPU_DIRECTER_CONFIG is set
echo $GPU_DIRECTER_CONFIG
# Should show: /home/user/.gpu-directer/rgpu.toml
# If empty, re-activate the environment:
conda deactivate && conda activate rgpu
```

### GPU not detected by Ollama

```bash
# Native Ollama:
nvidia-smi                        # Should show GPU
# If model runs slowly, check GPU is being used:
ollama run llama3.2 --verbose

# Docker Ollama:
docker exec ollama nvidia-smi    # Should list your GPU
# If this fails:
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
gpu-directer server restart
```

### Queue full or requests timing out

```bash
gpu-directer client status        # Shows current queue depth
# Increase timeout:
gpu-directer config set client.timeout_seconds=600
gpu-directer config set server.queue_timeout=600
```

### Model-not-found warning

```
UserWarning: Model 'llama3.2' not found on remote server, routing to local Ollama.
```

Pull the model on the server:
```bash
# Native Ollama (SSH into server):
ollama pull llama3.2

# Docker Ollama (SSH into server):
docker exec ollama ollama pull llama3.2
```
