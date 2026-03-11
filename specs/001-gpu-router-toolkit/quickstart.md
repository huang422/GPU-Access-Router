# Quickstart: GPU Access Router Toolkit

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11

End-to-end guide from zero to a working remote GPU inference call. Covers Tailscale setup, server install, client install, and first API call.

---

## Prerequisites

| Machine | Requirements |
|---|---|
| GPU Server | Ubuntu 20.04+, NVIDIA GPU, NVIDIA drivers installed, Docker installed |
| Client | macOS 12+ or Ubuntu 20.04+, Python 3.8+, pip |

---

## Part 1: Tailscale Network Setup (both machines)

Tailscale creates a private network between your machines, allowing the client to reach the GPU server even behind a mobile hotspot.

### 1.1 Create a Tailscale account

Go to [tailscale.com](https://tailscale.com) and sign up for a free account.

### 1.2 Install Tailscale on the GPU server (Ubuntu)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Follow the URL printed in the terminal to authenticate this device in your Tailscale account.

### 1.3 Install Tailscale on the client machine

- **macOS**: Download from [tailscale.com/download](https://tailscale.com/download) or `brew install tailscale`
- **Ubuntu**: Same as server: `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`

### 1.4 Get the GPU server's Tailscale IP

On the GPU server, run:
```bash
tailscale ip -4
# Outputs something like: 100.64.0.5
```

Save this IP — you'll need it during client setup.

### 1.5 Verify connectivity

From the client machine, check you can reach the server:
```bash
ping -c 3 100.64.0.5   # Use your actual Tailscale IP
```

If you see replies, Tailscale is working.

---

## Part 2: GPU Server Setup

### 2.1 Install GPU Access Router server

On the GPU server:
```bash
pip install "git+https://github.com/your-account/GPU-Access-Router-toolkit.git[server]"
```

### 2.2 Run the setup wizard

```bash
gpu-access-router server setup
```

The wizard will:
- Check Docker and NVIDIA drivers
- Pull and start the `ollama/ollama` Docker container with GPU access
- Configure Ollama to accept connections from the Tailscale network
- Write server configuration to `~/.gpu-access-router/config.toml`
- Print your Tailscale IP and confirmation that the server is ready

### 2.3 Pull at least one model

```bash
docker exec ollama ollama pull llama3.2
```

### 2.4 Verify server health

```bash
gpu-access-router server doctor
```

All checks should show `[✓]`. If any fail, follow the printed fix hints.

---

## Part 3: Client Setup

### 3.1 Install GPU Access Router client

On your laptop or cloud machine:
```bash
pip install "git+https://github.com/your-account/GPU-Access-Router-toolkit.git[client]"
```

### 3.2 Run the setup wizard

```bash
gpu-access-router client setup
```

When prompted, enter the GPU server's Tailscale IP (from step 1.4).

The wizard will:
- Check Tailscale is running on this machine
- Test connectivity to the server
- List all available models on the server
- Ask your preferred routing mode (press Enter for `auto`)
- Save configuration to `~/.gpu-access-router/config.toml`

### 3.3 Verify client status

```bash
gpu-access-router client status
```

You should see the server listed as `● online` with available models.

---

## Part 4: Use GPURouter in Your Project

### 4.1 Install in your project

```bash
# In your project's virtual environment
pip install "git+https://github.com/your-account/GPU-Access-Router-toolkit.git[client]"
```

### 4.2 Basic usage

```python
from gpu_access_router import GPURouter

router = GPURouter()

response = router.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}]
)

print(response.message.content)
```

### 4.3 Force a specific routing target

```python
# Always use remote GPU (error if unreachable)
response = router.chat(model="llama3.2", messages=msgs, prefer="remote")

# Always use local Ollama (ignore remote server)
response = router.chat(model="llama3.2", messages=msgs, prefer="local")

# Auto: remote first, local fallback (default)
response = router.chat(model="llama3.2", messages=msgs)
```

### 4.4 Check what's available

```python
models = router.list_models()
print("Remote models:", [m["name"] for m in models["remote"] or []])
print("Local models:", [m["name"] for m in models["local"] or []])

status = router.status()
print("Queue depth:", status["remote"]["queue_depth"])
```

---

## Part 5: Common Configuration Tasks

### Change routing mode permanently

```bash
# Use remote only
gpu-access-router config set client.routing_mode=remote

# Use local only
gpu-access-router config set client.routing_mode=local

# Auto (default)
gpu-access-router config set client.routing_mode=auto
```

### Update server IP (after Tailscale IP changes)

```bash
gpu-access-router config set client.server_ip=100.64.0.99
```

### Change queue timeout

```bash
# On client: how long to wait in server queue
gpu-access-router config set client.timeout_seconds=600

# On server: how long before dropping a queued request
gpu-access-router config set server.queue_timeout=600
```

### View all current configuration

```bash
gpu-access-router config show
```

### Edit config file directly

```bash
gpu-access-router config edit   # Opens ~/.gpu-access-router/config.toml in $EDITOR
```

---

## Part 6: Troubleshooting

### Server unreachable from client

```bash
# Check Tailscale is running on both machines
tailscale status

# Check server health
gpu-access-router server doctor   # run on server

# Check client connectivity
gpu-access-router client status   # run on client
```

### GPU not detected by Ollama

```bash
# On server:
gpu-access-router server doctor   # Look for [✗] GPU passthrough
docker exec ollama nvidia-smi   # Should show your GPU
```

If `nvidia-smi` fails inside the container, install `nvidia-container-toolkit`:
```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Queue full or requests timing out

```bash
# Check current queue
gpu-access-router client status   # Shows queue depth

# Increase timeout if requests are slow
gpu-access-router config set client.timeout_seconds=600
```

### Model not found warning

If you see `"Model X not found on remote server, routing to local Ollama"`:
```bash
# Pull the model on the server
docker exec ollama ollama pull <model-name>
```

---

## Reference: Complete Command List

```bash
# Server commands
gpu-access-router server setup        # Initial setup wizard
gpu-access-router server doctor       # Health check (all components)
gpu-access-router server models       # List available models
gpu-access-router server start        # Start Ollama container
gpu-access-router server stop         # Stop Ollama container
gpu-access-router server restart      # Restart Ollama container

# Client commands
gpu-access-router client setup        # Connect to server wizard
gpu-access-router client status       # Show server + local status
gpu-access-router client models       # List models (remote/local/all)

# Config commands
gpu-access-router config show         # Display all settings
gpu-access-router config set k=v      # Update one setting
gpu-access-router config edit         # Open config in $EDITOR
gpu-access-router config reset        # Reset to defaults

# Global
gpu-access-router --version           # Package version
gpu-access-router --help              # Help
gpu-access-router <command> --json    # Machine-readable output
gpu-access-router <command> --quiet   # Suppress info output
```
