"""ollama CLI shim — proxies read commands to the remote GPU server when configured."""

import json
import os
import subprocess
import sys
import urllib.error
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


# ---------------------------------------------------------------------------
# Remote proxied commands
# ---------------------------------------------------------------------------

def _cmd_list(server_ip: str, server_port: int) -> int:
    """ollama list — show remote models."""
    url = f"http://{server_ip}:{server_port}/gd/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"[gpu-directer] Cannot reach remote server: {exc}", file=sys.stderr)
        return 1

    models = data.get("models", [])
    print(f"NAME{'':40}ID{'':14}SIZE      MODIFIED")
    for m in models:
        name = m.get("name", "")
        model_id = (m.get("digest") or "")[:12]
        size = m.get("size", 0)
        size_str = _fmt_size(size)
        modified = (m.get("modified_at") or "")[:19].replace("T", " ")
        print(f"{name:<44}{model_id:<16}{size_str:<10}{modified}")

    if not models:
        print("(no models on remote server)")
    return 0


def _cmd_ps(server_ip: str, server_port: int) -> int:
    """ollama ps — show remote queue/processing status."""
    try:
        url = f"http://{server_ip}:{server_port}/gd/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            health = json.loads(resp.read())
        url2 = f"http://{server_ip}:{server_port}/gd/queue"
        with urllib.request.urlopen(url2, timeout=5) as resp:
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


def _cmd_pull_remote(model: str, server_ip: str) -> int:
    """ollama pull — instruct user to pull on the server."""
    print(f"[gpu-directer] 'ollama pull' downloads to the machine you run it on.")
    print(f"  To pull '{model}' on the remote GPU server, SSH in and run:")
    print(f"")
    print(f"    ssh <user>@{server_ip}")
    print(f"    ollama pull {model}")
    print(f"")
    print(f"  Or if using Docker on the server:")
    print(f"    docker exec ollama ollama pull {model}")
    return 0


# ---------------------------------------------------------------------------
# Fallthrough to real ollama
# ---------------------------------------------------------------------------

def _find_real_ollama() -> Optional[str]:
    """Find the real ollama binary, skipping this shim's directory."""
    import shutil
    shim_dir = str(Path(sys.argv[0]).resolve().parent) if sys.argv[0] else ""

    original_path = os.environ.get("PATH", "")
    filtered_dirs = [
        d for d in original_path.split(os.pathsep)
        if d and Path(d).resolve() != Path(shim_dir)
    ]
    filtered_path = os.pathsep.join(filtered_dirs)

    env = os.environ.copy()
    env["PATH"] = filtered_path
    real = shutil.which("ollama", path=filtered_path)
    return real


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

    # Commands we proxy to remote when configured
    if remote_active and server_ip:
        if subcmd in ("list", "ls"):
            sys.exit(_cmd_list(server_ip, server_port))

        if subcmd == "ps":
            sys.exit(_cmd_ps(server_ip, server_port))

        if subcmd == "pull" and len(args) >= 2:
            sys.exit(_cmd_pull_remote(args[1], server_ip))

    # Everything else: fall through to real ollama
    sys.exit(_passthrough(args))


if __name__ == "__main__":
    main()
