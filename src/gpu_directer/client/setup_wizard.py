"""Interactive client setup wizard."""

import sys
from typing import Optional

from rich.console import Console

from gpu_directer.core.constants import DEFAULT_PORT, DEFAULT_ROUTING_MODE, DEFAULT_TIMEOUT

console = Console()
err_console = Console(stderr=True)


def _ok(msg: str):
    console.print(f"  [green]✓[/green] {msg}")


def _fail(msg: str, hint: str = ""):
    err_console.print(f"  [red]✗[/red] {msg}")
    if hint:
        err_console.print(f"    [yellow]Fix:[/yellow] {hint}")


def run_client_setup(
    server_ip: Optional[str] = None,
    port: int = DEFAULT_PORT,
    non_interactive: bool = False,
    config_path: Optional[str] = None,
) -> None:
    """Execute client setup wizard."""
    from gpu_directer.client.connectivity import (
        check_tailscale_connected,
        check_tailscale_installed,
        probe_server,
        query_server_models,
    )

    console.print("\n[bold]GPU Directer — Client Setup Wizard[/bold]\n")

    # Step 1: Check Tailscale installed
    console.print("[bold cyan]Step 1/8:[/bold cyan] Checking Tailscale installation…")
    if not check_tailscale_installed():
        _fail(
            "Tailscale not installed.",
            "Install: curl -fsSL https://tailscale.com/install.sh | sh",
        )
        if non_interactive:
            sys.exit(1)
        # Allow user to continue without Tailscale for local-only mode
        if not _confirm("Continue without Tailscale (local routing only)?"):
            sys.exit(1)
    else:
        _ok("Tailscale is installed.")

        # Step 2: Check Tailscale connected
        console.print("[bold cyan]Step 2/8:[/bold cyan] Checking Tailscale connection…")
        ts = check_tailscale_connected()
        if not ts["connected"]:
            _fail("Tailscale not connected.", "Run: sudo tailscale up")
            if non_interactive:
                sys.exit(1)
            if not _confirm("Continue anyway?"):
                sys.exit(1)
        else:
            _ok(f"Tailscale connected. Your IP: {ts.get('own_ip', 'unknown')}")

    # Step 3: Get server IP
    console.print("[bold cyan]Step 3/8:[/bold cyan] Server IP address")
    if not server_ip:
        if non_interactive:
            err_console.print("[red]Error:[/red] --server-ip required in non-interactive mode.")
            sys.exit(1)
        server_ip = input("  Enter server Tailscale IP: ").strip()
        if not server_ip:
            err_console.print("[red]Error:[/red] Server IP is required.")
            sys.exit(1)
    _ok(f"Using server IP: {server_ip}")

    # Step 4: TCP probe
    console.print(f"[bold cyan]Step 4/8:[/bold cyan] Probing {server_ip}:{port}…")
    if not probe_server(server_ip, port, timeout=5):
        _fail(
            f"Cannot reach {server_ip}:{port}.",
            "Check Tailscale connection, server is running, and firewall rules.",
        )
        if non_interactive:
            sys.exit(2)
        if not _confirm("Continue anyway?"):
            sys.exit(2)
    else:
        _ok(f"Server at {server_ip}:{port} is reachable.")

    # Step 5: Query models
    console.print("[bold cyan]Step 5/8:[/bold cyan] Fetching available models…")
    models = query_server_models(server_ip, port)
    if models:
        _ok(f"Available models: {', '.join(models)}")
    else:
        console.print("  [yellow]⚠[/yellow] Could not fetch model list (server may be warming up).")

    # Step 6: Routing mode
    console.print("[bold cyan]Step 6/8:[/bold cyan] Routing mode")
    routing_mode = DEFAULT_ROUTING_MODE
    if not non_interactive:
        console.print("  Options: [bold]auto[/bold] (default), remote, local")
        choice = input(f"  Routing mode [{DEFAULT_ROUTING_MODE}]: ").strip()
        if choice in ("auto", "remote", "local"):
            routing_mode = choice
        elif choice:
            console.print(f"  [yellow]Unknown mode '{choice}', using '{DEFAULT_ROUTING_MODE}'.[/yellow]")
    _ok(f"Routing mode: {routing_mode}")

    # Step 7: Write config
    console.print("[bold cyan]Step 7/8:[/bold cyan] Saving configuration…")
    from gpu_directer import config as cfg_mod
    cfg = cfg_mod.load_config(config_path)
    cfg.setdefault("client", {})
    cfg["client"]["server_ip"] = server_ip
    cfg["client"]["server_port"] = port
    cfg["client"]["routing_mode"] = routing_mode
    cfg["client"]["timeout_seconds"] = DEFAULT_TIMEOUT
    cfg.setdefault("meta", {})["role"] = "client"
    cfg_mod.save_config(cfg, config_path)
    _ok("Configuration saved.")

    # Step 8: Print summary
    console.print("[bold cyan]Step 8/8:[/bold cyan] Done!\n")
    console.print("[bold green]✓ Client setup complete![/bold green]\n")
    console.print("Quick start:")
    console.print("  [bold]from gpu_directer import GPURouter[/bold]")
    console.print("  [bold]router = GPURouter()[/bold]")
    console.print('  [bold]response = router.chat("llama3.2", [{"role": "user", "content": "Hello!"}])[/bold]')
    console.print("  [bold]print(response.message.content)[/bold]")
    console.print("\nCheck status: [bold]gpu-directer client status[/bold]")


def _confirm(prompt: str) -> bool:
    answer = input(f"  {prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")
