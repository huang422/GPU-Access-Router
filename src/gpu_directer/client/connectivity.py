"""Client-side connectivity helpers."""

import json
import socket
import subprocess
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from gpu_directer.core.constants import DEFAULT_PORT


def check_tailscale_installed() -> bool:
    try:
        subprocess.run(
            ["tailscale", "--version"],
            capture_output=True, check=True, timeout=5,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_tailscale_connected() -> Dict:
    """Return {connected: bool, own_ip: str|None}."""
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"connected": False, "own_ip": None}
        data = json.loads(result.stdout)
        if data.get("BackendState") == "Running":
            self_node = data.get("Self", {})
            ips = self_node.get("TailscaleIPs", [])
            ipv4 = next((ip for ip in ips if "." in ip), None)
            return {"connected": True, "own_ip": ipv4}
        return {"connected": False, "own_ip": None}
    except FileNotFoundError:
        return {"connected": False, "own_ip": None}
    except Exception:
        return {"connected": False, "own_ip": None}


def probe_server(ip: str, port: int, timeout: int = 5) -> bool:
    """TCP probe — returns True if server is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def query_server_models(ip: str, port: int) -> Optional[List[str]]:
    """Return list of model names from /gd/models, or None on failure."""
    try:
        url = f"http://{ip}:{port}/gd/models"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return None


def query_server_health(ip: str, port: int) -> Optional[Dict]:
    """Return health dict from /gd/health, or None on failure."""
    try:
        url = f"http://{ip}:{port}/gd/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def query_local_models() -> Optional[List[str]]:
    """Return list of local Ollama model names, or None if unavailable."""
    try:
        with urllib.request.urlopen(f"http://localhost:{DEFAULT_PORT}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return None
