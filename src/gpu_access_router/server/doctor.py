"""Server diagnostic checks."""

import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List

from gpu_access_router.core.constants import DEFAULT_API_PORT, DEFAULT_PORT


def _check(name: str, status: str, detail: str, fix_hint: str = "") -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "fix_hint": fix_hint}


def check_ollama_running(port: int = DEFAULT_PORT) -> Dict[str, str]:
    """Check if the native Ollama service is responding."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/", timeout=5
        ) as resp:
            body = resp.read().decode()
            if "Ollama" in body:
                return _check("ollama_running", "pass", f"Ollama is running on port {port}")
        return _check("ollama_running", "pass", f"Service responding on port {port}")
    except Exception as exc:
        return _check(
            "ollama_running", "fail",
            f"Ollama not reachable on port {port}: {exc}",
            "Start Ollama: ollama serve",
        )


def check_gpu() -> Dict[str, str]:
    """Check if nvidia-smi is available on the host."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            gpu_name = "GPU detected"
            for line in lines:
                if "GeForce" in line or "RTX" in line or "GTX" in line or "Tesla" in line or "A100" in line:
                    gpu_name = line.strip().split("|")[1].strip() if "|" in line else line.strip()
                    break
            return _check("gpu", "pass", gpu_name)
        return _check(
            "gpu", "fail",
            "nvidia-smi failed",
            "Install NVIDIA drivers: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/",
        )
    except FileNotFoundError:
        return _check("gpu", "fail", "nvidia-smi not found", "Install NVIDIA drivers.")
    except Exception as exc:
        return _check("gpu", "fail", str(exc), "Check NVIDIA drivers.")


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
            "Pull a model: ollama pull llama3.2",
        )
    except Exception as exc:
        return _check(
            "ollama_models_available", "fail",
            f"Could not reach Ollama: {exc}",
            "Ensure Ollama is running: ollama serve",
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
            f"GPU Access Router server API not reachable: {exc}",
            "Start the server: gpu-access-router server serve",
        )


def run_doctor(ollama_port: int = DEFAULT_PORT, api_port: int = DEFAULT_API_PORT) -> Dict[str, Any]:
    """Run all diagnostic checks and return DiagnosticReport dict."""
    checks: List[Dict] = [
        check_ollama_running(ollama_port),
        check_gpu(),
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
