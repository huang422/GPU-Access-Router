"""Interactive server setup wizard."""

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from rich.console import Console

from gpu_access_router.core.constants import DEFAULT_API_PORT, DEFAULT_PORT, DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT

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
    api_port: int = DEFAULT_API_PORT,
    non_interactive: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Execute 8-step server setup wizard."""
    total = 8
    console.print("\n[bold]GPU Access Router — Server Setup Wizard[/bold]\n")

    # Step 1: Detect NVIDIA drivers
    _step(1, total, "Checking NVIDIA drivers…")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        _fail(
            "nvidia-smi not found or failed.",
            "Install NVIDIA drivers.",
        )
        if not non_interactive:
            if not _confirm("Continue anyway?"):
                sys.exit(1)
    else:
        _ok("NVIDIA GPU detected.")

    # Step 2: Check if Ollama is installed
    _step(2, total, "Checking Ollama installation…")
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        _fail("Ollama not found.", "Install Ollama: https://ollama.com/download")
        sys.exit(1)
    _ok(f"Ollama found: {result.stdout.strip()}")

    # Step 3: Check if Ollama is running
    _step(3, total, "Waiting for Ollama to be ready (up to 60s)…")
    if not _wait_for_ollama(port, timeout=60):
        _fail("Ollama did not become ready in time.", "Start Ollama: ollama serve")
        sys.exit(2)
    _ok("Ollama is ready.")

    # Step 4: Check / install Tailscale
    _step(4, total, "Checking Tailscale…")
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

    # Step 5: Prompt to run tailscale up
    _step(5, total, "Tailscale authentication")
    console.print("  Run the following command in another terminal:")
    console.print("    [bold]sudo tailscale up[/bold]")
    if not non_interactive:
        input("  Press Enter once Tailscale is authenticated… ")
    else:
        console.print("  [dim](non-interactive: skipping Tailscale auth prompt)[/dim]")

    # Step 6: Get Tailscale IPv4
    _step(6, total, "Reading Tailscale IP…")
    tailscale_ip = _get_tailscale_ip()
    if tailscale_ip:
        _ok(f"Tailscale IP: [bold]{tailscale_ip}[/bold]")
    else:
        _fail("Could not determine Tailscale IP.", "Run: tailscale ip -4")
        tailscale_ip = ""

    # Step 7: Write config
    _step(7, total, "Writing server configuration…")
    from gpu_access_router import config as cfg_mod
    cfg = cfg_mod.load_config(config_path)
    cfg.setdefault("server", {})
    cfg["server"]["ollama_port"] = port
    cfg["server"]["api_port"] = api_port
    cfg["server"]["queue_timeout"] = DEFAULT_TIMEOUT
    cfg["server"]["max_queue_depth"] = DEFAULT_QUEUE_DEPTH
    cfg.setdefault("meta", {})["role"] = "server"
    cfg_mod.save_config(cfg, config_path)
    _ok("Configuration saved.")

    # Step 8: Auto-start API server in background
    _step(8, total, "Starting GPU Access Router API server in background…")
    from gpu_access_router.cli import _start_api_server
    _start_api_server(api_port)

    console.print("\n[bold green]✓ Server setup complete![/bold green]")
    if tailscale_ip:
        console.print(f"\n[bold]Your Tailscale IP:[/bold] [cyan]{tailscale_ip}[/cyan]")
        console.print("Share this IP with your clients so they can connect.")
    console.print("\nNext steps:")
    console.print("  1. Pull a model: [bold]ollama pull llama3.2[/bold]")
    console.print("  2. Verify setup: [bold]gpu-access-router server doctor[/bold]")
    console.print(f"  3. On each client: [bold]gpu-access-router client setup --server-ip {tailscale_ip or '<IP>'} --port {api_port}[/bold]")
    console.print("\nManage the API server:")
    console.print("  gpu-access-router server start    # start in background")
    console.print("  gpu-access-router server stop     # stop")
    console.print("  gpu-access-router server restart  # restart")


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
