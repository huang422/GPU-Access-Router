# Contract: CLI Command Schema

**Branch**: `001-gpu-router-toolkit` | **Date**: 2026-03-11
**Entrypoint**: `gpu-directer` (installed via `pip install gpu-directer[...]`)

This document defines the complete command surface for the `gpu-directer` CLI. All commands follow the pattern: `gpu-directer <role> <command> [options]`. Output is human-readable by default; `--json` flag available on all commands for machine-readable output.

---

## Top-level

```
gpu-directer [--version] [--help]
```

| Flag | Description |
|---|---|
| `--version` | Print package version and exit |
| `--help` | Show command list and exit |

---

## `gpu-directer server` group

Commands for the GPU server machine (requires `[server]` extra installed).

### `gpu-directer server setup`

Interactive wizard to configure the GPU server from scratch.

```
gpu-directer server setup [--non-interactive] [--port PORT]
```

| Option | Default | Description |
|---|---|---|
| `--non-interactive` | false | Skip prompts; use defaults or fail if required info missing |
| `--port` | 11434 | Ollama port to expose |

**Wizard steps** (interactive mode):
1. Check Docker installed → print fix hint if missing, exit
2. Check NVIDIA drivers → print fix hint if missing, exit
3. Pull `ollama/ollama` Docker image
4. Start Ollama container with `--gpus all` and `OLLAMA_HOST=0.0.0.0`
5. Install Tailscale if not present
6. Prompt user to authenticate Tailscale (`tailscale up`)
7. Display Tailscale IP for user to share with clients
8. Write `[server]` section to `~/.gpu-directer/config.toml`
9. Print summary and next steps

**Exit codes**: `0` success, `1` prerequisite missing, `2` Docker/Tailscale error

---

### `gpu-directer server doctor`

Extended diagnostic check of all server components.

```
gpu-directer server doctor [--json]
```

| Check | What it verifies |
|---|---|
| `docker_installed` | `docker --version` succeeds |
| `ollama_container_running` | `docker inspect ollama` → State.Running == true |
| `gpu_passthrough` | `docker exec ollama nvidia-smi` exits 0 |
| `tailscale_connected` | `tailscale status --json` BackendState == "Running" |
| `ollama_models_available` | `/api/tags` returns ≥1 model |
| `queue_status` | Server API `/health` returns queue depth |

**Output format (human-readable)**:
```
[✓] Docker installed         Docker 24.0.7
[✓] Ollama container running Container 'ollama' is running
[✓] GPU passthrough active   NVIDIA GeForce RTX 4090
[✗] Tailscale connected      BackendState: Stopped
    Fix: sudo tailscale up
[✓] Models available         3 models: llama3.2, mistral, codellama
[✓] Queue status             Depth: 0, Processing: idle

Overall: FAIL (1 check failed)
```

**Exit codes**: `0` all pass, `1` one or more checks failed

---

### `gpu-directer server models`

List all Ollama models available on the server.

```
gpu-directer server models [--json]
```

**Output (human-readable)**:
```
Available models on this server:
  llama3.2        3.8 GB   Q4_K_M
  mistral         4.1 GB   Q4_0
  codellama       3.5 GB   Q4_K_M
```

---

### `gpu-directer server start` / `stop` / `restart`

Manage the Ollama Docker container lifecycle.

```
gpu-directer server start
gpu-directer server stop
gpu-directer server restart
```

Each command runs the corresponding `docker start|stop|restart ollama` and reports success/failure.

---

## `gpu-directer client` group

Commands for the client machine (requires `[client]` extra installed).

### `gpu-directer client setup`

Interactive wizard to connect a client to the GPU server.

```
gpu-directer client setup [--server-ip IP] [--port PORT] [--non-interactive]
```

| Option | Default | Description |
|---|---|---|
| `--server-ip` | prompted | Tailscale IP of the GPU server |
| `--port` | 11434 | Ollama port on the server |
| `--non-interactive` | false | Use provided flags without prompting |

**Wizard steps**:
1. Check Tailscale installed on this machine
2. Check Tailscale connected (`tailscale status`)
3. Prompt for server Tailscale IP (if not `--server-ip`)
4. TCP probe to `server_ip:port` (5s timeout)
5. Query `/api/tags` on server → list available models
6. Prompt for default routing mode (`auto` / `remote` / `local`)
7. Write `[client]` section to `~/.gpu-directer/config.toml`
8. Print summary and quickstart code example

**Exit codes**: `0` success, `1` Tailscale not installed/connected, `2` server unreachable

---

### `gpu-directer client status`

Show current connection status between client and configured server.

```
gpu-directer client status [--json]
```

**Output (human-readable)**:
```
GPU Directer Client Status
  Server IP:      100.64.0.5:11434
  Server status:  ● online
  Queue depth:    2 requests waiting
  Routing mode:   auto
  Available models (remote):
    llama3.2, mistral, codellama
  Local Ollama:   ● online
    Local models: llama3.2
```

**Exit codes**: `0` at least one source reachable, `1` all sources unreachable

---

### `gpu-directer client models`

List models available on the remote server and/or local Ollama.

```
gpu-directer client models [--source remote|local|all] [--json]
```

| Option | Default | Description |
|---|---|---|
| `--source` | `all` | Which source to query |

---

## `gpu-directer config` group

Configuration management (available on all machines).

### `gpu-directer config show`

Print current configuration.

```
gpu-directer config show [--json]
```

**Output**: All `config.toml` sections printed as readable key-value pairs. Sensitive values (IPs) shown in full.

---

### `gpu-directer config set`

Update a single configuration value.

```
gpu-directer config set <key>=<value>
```

**Examples**:
```bash
gpu-directer config set client.server_ip=100.64.0.5
gpu-directer config set client.routing_mode=local
gpu-directer config set client.timeout_seconds=600
gpu-directer config set server.queue_timeout=120
```

**Validation**: Key must exist in schema; value must pass type validation. Prints error on invalid key or value.

---

### `gpu-directer config edit`

Open config file in `$EDITOR` (falls back to `nano` if unset).

```
gpu-directer config edit
```

Opens `~/.gpu-directer/config.toml` directly. No validation is performed after edit (user is responsible).

---

### `gpu-directer config reset`

Reset configuration to defaults (prompts for confirmation).

```
gpu-directer config reset [--yes]
```

| Option | Description |
|---|---|
| `--yes` | Skip confirmation prompt |

---

## Global Flags

Available on all commands:

| Flag | Description |
|---|---|
| `--json` | Output as JSON instead of human-readable text |
| `--quiet` | Suppress informational output; only print errors |
| `--config PATH` | Use a custom config file path instead of `~/.gpu-directer/config.toml` |

---

## Error Conventions

All errors are printed to stderr. Exit codes:
- `0` — success
- `1` — configuration or prerequisite error (user-fixable)
- `2` — connectivity or runtime error
- `3` — internal error (bug)

Error messages always include a "Fix:" hint where applicable.
