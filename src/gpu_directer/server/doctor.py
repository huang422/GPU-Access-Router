"""Server diagnostic checks."""

import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

from gpu_directer.core.constants import DEFAULT_API_PORT, DEFAULT_PORT


def _check(name: str, status: str, detail: str, fix_hint: str = "") -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "fix_hint": fix_hint}


def check_docker() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return _check("docker_installed", "pass", version)
        return _check(
            "docker_installed", "fail",
            "docker --version failed",
            "Install Docker: https://docs.docker.com/engine/install/",
        )
    except FileNotFoundError:
        return _check(
            "docker_installed", "fail",
            "Docker not found in PATH",
            "Install Docker: https://docs.docker.com/engine/install/",
        )
    except Exception as exc:
        return _check("docker_installed", "fail", str(exc), "Install Docker.")


def check_ollama_container() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", "ollama"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            return _check("ollama_container_running", "pass", "Container 'ollama' is running")
        return _check(
            "ollama_container_running", "fail",
            "Container 'ollama' is not running",
            "Run: docker start ollama",
        )
    except FileNotFoundError:
        return _check("ollama_container_running", "fail", "Docker not available", "Install Docker.")
    except Exception as exc:
        return _check("ollama_container_running", "fail", str(exc), "Run: docker start ollama")


def check_gpu_passthrough() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["docker", "exec", "ollama", "nvidia-smi"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            # Extract GPU name from nvidia-smi output
            lines = result.stdout.splitlines()
            gpu_name = "GPU detected"
            for line in lines:
                if "GeForce" in line or "RTX" in line or "GTX" in line or "Tesla" in line or "A100" in line:
                    gpu_name = line.strip().split("|")[1].strip() if "|" in line else line.strip()
                    break
            return _check("gpu_passthrough", "pass", gpu_name)
        return _check(
            "gpu_passthrough", "fail",
            "nvidia-smi failed inside container",
            "Install nvidia-container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html",
        )
    except FileNotFoundError:
        return _check("gpu_passthrough", "fail", "Docker not available", "Install Docker.")
    except Exception as exc:
        return _check("gpu_passthrough", "fail", str(exc), "Check nvidia-container-toolkit.")


def check_tailscale() -> Dict[str, str]:
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return _check(
                "tailscale_connected", "fail",
                f"tailscale exit code: {result.returncode}",
                "Run: sudo tailscale up",
            )
        data = json.loads(result.stdout)
        backend = data.get("BackendState", "")
        if backend == "Running":
            # Extract own IPv4
            self_node = data.get("Self", {})
            tailscale_ips = self_node.get("TailscaleIPs", [])
            ipv4 = next((ip for ip in tailscale_ips if "." in ip), "unknown")
            return _check("tailscale_connected", "pass", f"Connected. Tailscale IP: {ipv4}")
        return _check(
            "tailscale_connected", "fail",
            f"BackendState: {backend}",
            "Run: sudo tailscale up",
        )
    except FileNotFoundError:
        return _check(
            "tailscale_connected", "fail",
            "Tailscale not found in PATH",
            "Install: curl -fsSL https://tailscale.com/install.sh | sh",
        )
    except Exception as exc:
        return _check("tailscale_connected", "fail", str(exc), "Run: sudo tailscale up")


def check_ollama_models(port: int = DEFAULT_PORT) -> Dict[str, str]:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/api/tags", timeout=5
        ) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if models:
            names = ", ".join(m["name"] for m in models[:5])
            return _check(
                "ollama_models_available", "pass",
                f"{len(models)} model{'s' if len(models) != 1 else ''}: {names}",
            )
        return _check(
            "ollama_models_available", "fail",
            "No models pulled yet",
            "Pull a model: docker exec ollama ollama pull llama3.2",
        )
    except Exception as exc:
        return _check(
            "ollama_models_available", "fail",
            f"Could not reach Ollama: {exc}",
            "Ensure container is running: gpu-directer server start",
        )


def check_queue_status(api_port: int = DEFAULT_API_PORT) -> Dict[str, str]:
    try:
        with urllib.request.urlopen(
            f"http://localhost:{api_port}/gd/health", timeout=5
        ) as resp:
            data = json.loads(resp.read())
        depth = data.get("queue_depth", 0)
        processing = data.get("processing", False)
        state = "processing" if processing else "idle"
        return _check(
            "queue_status", "pass",
            f"Queue depth: {depth}, Processing: {state}",
        )
    except Exception as exc:
        return _check(
            "queue_status", "fail",
            f"GPU Directer server API not reachable: {exc}",
            "Start the server: gpu-directer server serve",
        )


def run_doctor(ollama_port: int = DEFAULT_PORT, api_port: int = DEFAULT_API_PORT) -> Dict[str, Any]:
    """Run all 6 diagnostic checks and return DiagnosticReport dict."""
    checks: List[Dict] = [
        check_docker(),
        check_ollama_container(),
        check_gpu_passthrough(),
        check_tailscale(),
        check_ollama_models(ollama_port),
        check_queue_status(api_port),
    ]
    overall = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": overall,
        "checks": checks,
    }
