# GPU Directer

Route LLM inference calls to a remote GPU server or fall back to local Ollama automatically — with a single Python class and a simple CLI.

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
 │  FastAPI +      │
 │  Serial Queue   │
 │  + Ollama GPU   │
 └─────────────────┘
```

## Two Roles

| Role | Machine | Purpose |
|---|---|---|
| **gpu-server** | Ubuntu + NVIDIA GPU | Runs Ollama in Docker, exposes queue API over Tailscale |
| **gpu-client** | macOS or Ubuntu (no GPU needed) | Connects to server, routes `GPURouter.chat()` calls |

## Requirements

| Machine | Requirements |
|---|---|
| GPU Server | Ubuntu 20.04+, NVIDIA GPU + drivers, Docker |
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

```bash
# Install (on the GPU server)
pip install "git+https://github.com/huang422/GPU-Directer.git[server]"

# Run the interactive setup wizard
gpu-directer server setup
```

The wizard:
- Checks Docker and NVIDIA drivers
- Pulls and starts `ollama/ollama` Docker container with `--gpus all`
- Installs Tailscale if needed and prompts you to authenticate
- Prints your Tailscale IP and writes `~/.gpu-directer/config.toml`

```bash
# Pull a model
docker exec ollama ollama pull llama3.2

# Verify everything is healthy
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

---

## Part 3: Client Setup

```bash
# Install (on your laptop / cloud machine)
pip install "git+https://github.com/your-account/GPU-Directer.git[client]"

# Run the interactive setup wizard
gpu-directer client setup
# → Enter the GPU server's Tailscale IP when prompted

# Verify the connection
gpu-directer client status
```

Expected output:
```
GPU Directer Client Status
  Server IP:      100.64.0.5:11434
  Server status:  ● online
  Queue depth:    0 requests waiting
  Routing mode:   auto
  Available models (remote):
    llama3.2
  Local Ollama:   ● offline
```

---

## Part 4: Use GPURouter in Your Code

```python
from gpu_directer import GPURouter

router = GPURouter()

response = router.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}]
)

print(response.message.content)
```

### Routing modes

```python
# Auto (default): remote first, fall back to local if unavailable
response = router.chat(model="llama3.2", messages=msgs)

# Force remote only (raises GPUDirecterConnectionError if unreachable)
response = router.chat(model="llama3.2", messages=msgs, prefer="remote")

# Force local only
response = router.chat(model="llama3.2", messages=msgs, prefer="local")
```

### Check what's available

```python
# List models
models = router.list_models()
print("Remote:", [m["name"] for m in models["remote"] or []])
print("Local:", [m["name"] for m in models["local"] or []])

# Full status
status = router.status()
print("Queue depth:", status["remote"].get("queue_depth", 0))
print("Routing mode:", status["routing_mode"])
```

---

## Part 5: Configuration Reference

Config is stored at `~/.gpu-directer/config.toml`:

```toml
[client]
server_ip       = "100.64.0.5"   # Tailscale IP of the GPU server
server_port     = 11434           # Ollama port (default: 11434)
routing_mode    = "auto"          # "auto" | "remote" | "local"
timeout_seconds = 300             # Queue wait timeout in seconds
default_model   = ""              # Optional: pre-select a model

[server]
ollama_port     = 11434           # Port Ollama container listens on
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

# Open in $EDITOR
gpu-directer config edit

# Reset to defaults
gpu-directer config reset
```

---

## Part 6: Complete Command Reference

### Server commands (requires `[server]` install)

| Command | Description |
|---|---|
| `gpu-directer server setup [--non-interactive] [--port PORT]` | Interactive setup wizard |
| `gpu-directer server doctor [--json]` | 6-point health check |
| `gpu-directer server models [--json]` | List models on this server |
| `gpu-directer server start` | Start Ollama Docker container |
| `gpu-directer server stop` | Stop Ollama Docker container |
| `gpu-directer server restart` | Restart Ollama Docker container |

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

## Troubleshooting

### Server unreachable from client

```bash
tailscale status              # Check Tailscale is running on both machines
gpu-directer server doctor    # Run on server — check all 6 items
gpu-directer client status    # Run on client — shows connectivity
```

### GPU not detected by Ollama

```bash
# On server:
docker exec ollama nvidia-smi   # Should list your GPU
# If this fails:
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
gpu-directer server restart
```

### Queue full or requests timing out

```bash
gpu-directer client status    # Shows current queue depth
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
docker exec ollama ollama pull llama3.2
```
