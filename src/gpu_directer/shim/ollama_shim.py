"""ollama CLI shim — proxies commands to the remote GPU server when configured.

Command routing:
  Remote proxied : list, ls, ps, show, run <model> <prompt>
  SSH guidance   : pull, push, rm, stop, cp, create, serve
  Fall through   : run <model> (interactive), help, anything unknown
"""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Remote detection
# ---------------------------------------------------------------------------

def _load_server_info():
    """Return (server_ip, server_port) if remote config is active, else (None, None)."""
    try:
        from gpu_directer import config as cfg_mod
        cfg = cfg_mod.load_config()
        client = cfg.get("client", {})
        routing_mode = client.get("routing_mode", "auto")
        server_ip = client.get("server_ip", "")
        server_port = int(client.get("server_port", 8080))
        if routing_mode == "local" or not server_ip:
            return None, None
        return server_ip, server_port
    except Exception:
        return None, None


def _is_remote_active(server_ip: Optional[str], server_port: int) -> bool:
    if not server_ip:
        return False
    try:
        url = f"http://{server_ip}:{server_port}/gd/health"
        with urllib.request.urlopen(url, timeout=3):
            return True
    except Exception:
        return False


def _fetch_remote_models(server_ip: str, server_port: int):
    url = f"http://{server_ip}:{server_port}/gd/models"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read()).get("models", [])


# ---------------------------------------------------------------------------
# Remote proxied commands
# ---------------------------------------------------------------------------

def _cmd_list(server_ip: str, server_port: int) -> int:
    """ollama list — show remote models."""
    try:
        models = _fetch_remote_models(server_ip, server_port)
    except Exception as exc:
        print(f"[gpu-directer] Cannot reach remote server: {exc}", file=sys.stderr)
        return 1

    print(f"NAME{'':40}ID{'':14}SIZE      MODIFIED")
    for m in models:
        name = m.get("name", "")
        model_id = (m.get("digest") or "")[:12]
        size_str = _fmt_size(m.get("size", 0))
        modified = (m.get("modified_at") or "")[:19].replace("T", " ")
        print(f"{name:<44}{model_id:<16}{size_str:<10}{modified}")

    if not models:
        print("(no models on remote server)")
    return 0


def _cmd_show(model: str, server_ip: str, server_port: int) -> int:
    """ollama show <model> — show remote model info."""
    try:
        models = _fetch_remote_models(server_ip, server_port)
    except Exception as exc:
        print(f"[gpu-directer] Cannot reach remote server: {exc}", file=sys.stderr)
        return 1

    match = next((m for m in models if m.get("name", "") == model), None)
    if match is None:
        # Try prefix match (e.g. "llama3.2" matches "llama3.2:latest")
        match = next((m for m in models if m.get("name", "").startswith(model)), None)

    if match is None:
        available = ", ".join(m.get("name", "") for m in models) or "none"
        print(f"[gpu-directer] Model '{model}' not found on remote server.", file=sys.stderr)
        print(f"  Available: {available}", file=sys.stderr)
        return 1

    name = match.get("name", "")
    digest = match.get("digest", "")
    size = _fmt_size(match.get("size", 0))
    modified = (match.get("modified_at") or "")[:19].replace("T", " ")
    family = (match.get("details") or {}).get("family", "")
    params = (match.get("details") or {}).get("parameter_size", "")
    quant = (match.get("details") or {}).get("quantization_level", "")

    print(f"  Model")
    print(f"    name:       {name}")
    print(f"    size:       {size}")
    print(f"    digest:     {digest}")
    print(f"    modified:   {modified}")
    if family:
        print(f"    family:     {family}")
    if params:
        print(f"    parameters: {params}")
    if quant:
        print(f"    quantization: {quant}")
    return 0


def _cmd_ps(server_ip: str, server_port: int) -> int:
    """ollama ps — show remote queue/processing status."""
    try:
        with urllib.request.urlopen(
            f"http://{server_ip}:{server_port}/gd/health", timeout=5
        ) as resp:
            health = json.loads(resp.read())
        with urllib.request.urlopen(
            f"http://{server_ip}:{server_port}/gd/queue", timeout=5
        ) as resp:
            queue = json.loads(resp.read())
    except Exception as exc:
        print(f"[gpu-directer] Cannot reach remote server: {exc}", file=sys.stderr)
        return 1

    processing = health.get("processing", False)
    queue_depth = health.get("queue_depth", 0)
    uptime = health.get("uptime_seconds", 0)

    print(f"Remote GPU server  {server_ip}:{server_port}")
    print(f"  Status:    {'processing' if processing else 'idle'}")
    print(f"  Queue:     {queue_depth} waiting")
    print(f"  Uptime:    {_fmt_uptime(uptime)}")

    requests = queue.get("requests", [])
    if requests:
        print()
        print(f"  {'REQUEST ID':38}{'POSITION':10}MODEL")
        for r in requests:
            print(f"  {r.get('request_id',''):<38}{r.get('position',0):<10}{r.get('model','')}")
    return 0


def _cmd_run_remote(model: str, prompt: str) -> int:
    """ollama run <model> <prompt> — proxy single-shot inference to remote GPU."""
    try:
        from gpu_directer.client.router import GPURouter
        router = GPURouter()
        response = router.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            prefer="remote",
        )
        print(response.message.content)
        return 0
    except Exception as exc:
        print(f"[gpu-directer] Remote inference failed: {exc}", file=sys.stderr)
        print("[gpu-directer] Falling back to local ollama...", file=sys.stderr)
        return _passthrough(["run", model, prompt])


def _cmd_ssh_only(subcmd: str, server_ip: str, extra_args: List[str]) -> int:
    """Commands that must run on the server — print SSH instructions."""
    model_arg = extra_args[0] if extra_args else "<model>"
    extra = " ".join(extra_args)

    DOCKER_CMDS = {
        "pull": f"docker exec ollama ollama pull {model_arg}",
        "rm":   f"docker exec ollama ollama rm {model_arg}",
        "stop": f"docker exec ollama ollama stop {model_arg}",
        "push": f"docker exec ollama ollama push {model_arg}",
        "cp":   f"docker exec ollama ollama cp {extra}",
        "create": f"docker exec ollama ollama create {extra}",
        "serve":  "# (ollama serve is already running on the server)",
    }
    native_cmd = f"ollama {subcmd} {extra}".strip()
    docker_cmd = DOCKER_CMDS.get(subcmd, native_cmd)

    print(f"[gpu-directer] '{subcmd}' must run on the remote GPU server.")
    print(f"  SSH in and run:")
    print(f"")
    print(f"    ssh <user>@{server_ip}")
    print(f"    {native_cmd}      # native Ollama")
    print(f"    {docker_cmd}      # Docker Ollama")
    return 0


# ---------------------------------------------------------------------------
# Fallthrough to real ollama
# ---------------------------------------------------------------------------

def _find_real_ollama() -> Optional[str]:
    """Find the real ollama binary, skipping this shim's directory."""
    import shutil
    shim_dir = str(Path(sys.argv[0]).resolve().parent) if sys.argv[0] else ""
    filtered_path = os.pathsep.join(
        d for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and Path(d).resolve() != Path(shim_dir)
    )
    return shutil.which("ollama", path=filtered_path)


def _passthrough(args: List[str]) -> int:
    """Exec the real ollama with the given args."""
    real = _find_real_ollama()
    if not real:
        print("[gpu-directer] Real 'ollama' binary not found in PATH.", file=sys.stderr)
        return 1
    result = subprocess.run([real] + args)
    return result.returncode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} TB"


def _fmt_uptime(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    subcmd = args[0] if args else ""

    server_ip, server_port = _load_server_info()
    remote_active = _is_remote_active(server_ip, server_port)

    if remote_active and server_ip:

        # --- Read commands: proxy to remote ---
        if subcmd in ("list", "ls"):
            sys.exit(_cmd_list(server_ip, server_port))

        if subcmd == "ps":
            sys.exit(_cmd_ps(server_ip, server_port))

        if subcmd == "show" and len(args) >= 2:
            sys.exit(_cmd_show(args[1], server_ip, server_port))

        # ollama run <model> <prompt>  — non-interactive single-shot
        if subcmd == "run" and len(args) >= 3:
            sys.exit(_cmd_run_remote(args[1], args[2]))

        # --- Write/manage commands: must run on server ---
        if subcmd in ("pull", "push", "rm", "stop", "cp", "create", "serve"):
            sys.exit(_cmd_ssh_only(subcmd, server_ip, args[1:]))

    # Everything else (interactive run, help, unknown) → fall through
    sys.exit(_passthrough(args))


if __name__ == "__main__":
    main()
