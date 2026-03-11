"""Interactive server setup wizard."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from rich.console import Console

from gpu_directer.core.constants import DEFAULT_PORT, DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT

console = Console()
err_console = Console(stderr=True)


def _step(n: int, total: int, msg: str):
    console.print(f"[bold cyan]Step {n}/{total}:[/bold cyan] {msg}")


def _ok(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def _fail(msg: str, hint: str = ""):
    err_console.print(f"  [red]✗[/red] {msg}")
    if hint:
        err_console.print(f"    [yellow]Fix:[/yellow] {hint}")


def run_server_setup(
    port: int = DEFAULT_PORT,
    non_interactive: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Execute 9-step server setup wizard."""
    total = 9
    console.print("\n[bold]GPU Directer — Server Setup Wizard[/bold]\n")

    # Step 1: Check Docker
    _step(1, total, "Checking Docker installation…")
    result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        _fail("Docker not found.", "Install Docker: https://docs.docker.com/engine/install/")
        sys.exit(1)
    _ok(f"Docker found: {result.stdout.strip()}")

    # Step 2: Detect NVIDIA drivers
    _step(2, total, "Checking NVIDIA drivers…")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        _fail(
            "nvidia-smi not found or failed.",
            "Install NVIDIA drivers and nvidia-container-toolkit.",
        )
        if not non_interactive:
            if not _confirm("Continue anyway?"):
                sys.exit(1)
    else:
        _ok("NVIDIA GPU detected.")

    # Step 3: Pull ollama/ollama Docker image
    _step(3, total, "Pulling ollama/ollama Docker image (this may take a while)…")
    result = subprocess.run(["docker", "pull", "ollama/ollama"], timeout=300)
    if result.returncode != 0:
        _fail("Failed to pull ollama/ollama image.", "Check your internet connection and Docker daemon.")
        sys.exit(2)
    _ok("ollama/ollama image pulled.")

    # Step 4: Start Ollama container
    _step(4, total, "Starting Ollama container with GPU access…")
    # Remove existing container if present
    subprocess.run(["docker", "rm", "-f", "ollama"], capture_output=True)
    run_cmd = [
        "docker", "run", "-d",
        "--gpus", "all",
        "--name", "ollama",
        "-e", f"OLLAMA_HOST=0.0.0.0:{port}",
        "-v", "ollama:/root/.ollama",
        "-p", f"{port}:{port}",
        "--restart", "unless-stopped",
        "ollama/ollama",
    ]
    result = subprocess.run(run_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _fail(f"Failed to start container: {result.stderr.strip()}", "Check Docker and GPU drivers.")
        sys.exit(2)
    _ok("Ollama container started.")

    # Step 5: Wait for Ollama ready
    _step(5, total, "Waiting for Ollama to be ready (up to 60s)…")
    if not _wait_for_ollama(port, timeout=60):
        _fail("Ollama did not become ready in time.", "Check: docker logs ollama")
        sys.exit(2)
    _ok("Ollama is ready.")

    # Step 6: Check / install Tailscale
    _step(6, total, "Checking Tailscale…")
    result = subprocess.run(["tailscale", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("  Tailscale not found. Installing…")
        result = subprocess.run(
            "curl -fsSL https://tailscale.com/install.sh | sh",
            shell=True,
        )
        if result.returncode != 0:
            _fail("Failed to install Tailscale.", "Install manually: https://tailscale.com/download")
            sys.exit(2)
    _ok("Tailscale is installed.")

    # Step 7: Prompt to run tailscale up
    _step(7, total, "Tailscale authentication")
    console.print("  Run the following command in another terminal:")
    console.print("    [bold]sudo tailscale up[/bold]")
    if not non_interactive:
        input("  Press Enter once Tailscale is authenticated… ")
    else:
        console.print("  [dim](non-interactive: skipping Tailscale auth prompt)[/dim]")

    # Step 8: Get Tailscale IPv4
    _step(8, total, "Reading Tailscale IP…")
    tailscale_ip = _get_tailscale_ip()
    if tailscale_ip:
        _ok(f"Tailscale IP: [bold]{tailscale_ip}[/bold]")
    else:
        _fail("Could not determine Tailscale IP.", "Run: tailscale ip -4")
        tailscale_ip = ""

    # Step 9: Write config
    _step(9, total, "Writing server configuration…")
    from gpu_directer import config as cfg_mod
    cfg = cfg_mod.load_config(config_path)
    cfg.setdefault("server", {})
    cfg["server"]["ollama_port"] = port
    cfg["server"]["queue_timeout"] = DEFAULT_TIMEOUT
    cfg["server"]["max_queue_depth"] = DEFAULT_QUEUE_DEPTH
    cfg.setdefault("meta", {})["role"] = "server"
    cfg_mod.save_config(cfg, config_path)
    _ok("Configuration saved.")

    console.print("\n[bold green]✓ Server setup complete![/bold green]")
    if tailscale_ip:
        console.print(f"\n[bold]Your Tailscale IP:[/bold] [cyan]{tailscale_ip}[/cyan]")
        console.print("Share this IP with your clients so they can connect.")
    console.print("\nNext steps:")
    console.print("  1. Pull a model: [bold]docker exec ollama ollama pull llama3.2[/bold]")
    console.print("  2. Verify setup: [bold]gpu-directer server doctor[/bold]")
    console.print(f"  3. On each client: [bold]gpu-directer client setup --server-ip {tailscale_ip or '<IP>'}[/bold]")


def _wait_for_ollama(port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/api/tags", timeout=3
            ):
                return True
        except Exception:
            time.sleep(2)
    return False


def _get_tailscale_ip() -> Optional[str]:
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            self_node = data.get("Self", {})
            ips = self_node.get("TailscaleIPs", [])
            return next((ip for ip in ips if "." in ip), None)
    except Exception:
        pass
    return None


def _confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")
