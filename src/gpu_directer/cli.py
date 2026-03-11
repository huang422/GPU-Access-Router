"""GPU Directer CLI — entry point for all sub-commands."""

import json
import os
import subprocess
import sys

import click
from rich.console import Console

import gpu_directer
from gpu_directer import __version__
from gpu_directer.core.constants import DEFAULT_API_PORT, DEFAULT_PORT

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shared context helpers
# ---------------------------------------------------------------------------

class _Ctx:
    """Simple object to carry shared state through Click's context."""
    def __init__(self, config_path, json_output, quiet):
        self.config_path = config_path
        self.json_output = json_output
        self.quiet = quiet


pass_ctx = click.make_pass_decorator(_Ctx, ensure=True)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="gpu-directer")
@click.option("--config", "config_path", default=None, metavar="PATH",
              help="Path to config.toml (default: ~/.gpu-directer/config.toml).")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Output as JSON instead of human-readable text.")
@click.option("--quiet", is_flag=True, default=False,
              help="Suppress informational output; only print errors.")
@click.pass_context
def main(ctx, config_path, json_output, quiet):
    """GPU Directer — route LLM inference to a remote GPU server or local Ollama."""
    ctx.ensure_object(dict)
    ctx.obj = _Ctx(config_path=config_path, json_output=json_output, quiet=quiet)


# ---------------------------------------------------------------------------
# server group
# ---------------------------------------------------------------------------

@main.group()
@click.option("--json", "json_output", is_flag=True, default=False)
@click.option("--quiet", is_flag=True, default=False)
@click.pass_context
def server(ctx, json_output, quiet):
    """Commands for the GPU server machine (requires [server] extra)."""
    parent: _Ctx = ctx.obj
    if json_output:
        parent.json_output = True
    if quiet:
        parent.quiet = True


def _require_server_deps():
    from gpu_directer.server import HAVE_SERVER
    if not HAVE_SERVER:
        err_console.print(
            "[red]Server dependencies not installed. Install with:[/red]\n"
            "  pip install gpu-directer[server]"
        )
        sys.exit(1)


@server.command("setup")
@click.option("--non-interactive", is_flag=True, default=False)
@click.option("--port", default=DEFAULT_PORT, type=int, show_default=True, help="Ollama port.")
@click.option("--api-port", default=DEFAULT_API_PORT, type=int, show_default=True, help="GPU Directer API port.")
@click.pass_context
def server_setup(ctx, non_interactive, port, api_port):
    """Interactive wizard to configure the GPU server from scratch."""
    _require_server_deps()
    from gpu_directer.server.setup_wizard import run_server_setup
    run_server_setup(port=port, api_port=api_port, non_interactive=non_interactive, config_path=ctx.obj.config_path)


@server.command("doctor")
@click.option("--json", "json_flag", is_flag=True, default=False)
@click.pass_context
def server_doctor(ctx, json_flag):
    """Extended diagnostic check of all server components."""
    _require_server_deps()
    from gpu_directer.server.doctor import run_doctor
    from gpu_directer import config as cfg_mod
    _cfg = cfg_mod.load_config(ctx.obj.config_path)
    _ollama_port = _cfg.get("server", {}).get("ollama_port", DEFAULT_PORT)
    _api_port = _cfg.get("server", {}).get("api_port", DEFAULT_API_PORT)
    report = run_doctor(ollama_port=_ollama_port, api_port=_api_port)
    obj: _Ctx = ctx.obj
    if obj.json_output or json_flag:
        click.echo(json.dumps(report, indent=2))
    else:
        _print_doctor_report(report)
    overall = report.get("overall", "fail")
    sys.exit(0 if overall == "pass" else 1)


def _print_doctor_report(report):
    for check in report.get("checks", []):
        status_icon = "[green][✓][/green]" if check["status"] == "pass" else "[red][✗][/red]"
        name = check["name"].replace("_", " ").title()
        detail = check.get("detail", "")
        console.print(f"{status_icon} {name:<28} {detail}")
        if check["status"] != "pass" and check.get("fix_hint"):
            console.print(f"    [yellow]Fix:[/yellow] {check['fix_hint']}")
    overall = report.get("overall", "fail")
    failed = sum(1 for c in report.get("checks", []) if c["status"] != "pass")
    if overall == "pass":
        console.print("\n[green]Overall: PASS[/green]")
    else:
        console.print(f"\n[red]Overall: FAIL ({failed} check{'s' if failed != 1 else ''} failed)[/red]")


@server.command("models")
@click.option("--json", "json_flag", is_flag=True, default=False)
@click.pass_context
def server_models(ctx, json_flag):
    """List all Ollama models available on this server."""
    _require_server_deps()
    import urllib.request
    import urllib.error
    obj: _Ctx = ctx.obj
    from gpu_directer import config as cfg_mod
    config = cfg_mod.load_config(obj.config_path)
    port = config.get("server", {}).get("api_port", DEFAULT_API_PORT)
    url = f"http://localhost:{port}/gd/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        err_console.print(f"[red]Failed to reach server API: {exc}[/red]")
        sys.exit(2)
    if obj.json_output or json_flag:
        click.echo(json.dumps(data, indent=2))
        return
    models = data.get("models", [])
    if not models:
        console.print("No models available on this server.")
        return
    console.print("Available models on this server:")
    for m in models:
        size_gb = m.get("size", 0) / 1e9
        quant = m.get("details", {}).get("quantization_level", "")
        console.print(f"  {m['name']:<20} {size_gb:.1f} GB   {quant}")


@server.command("start")
def server_start():
    """Start the Ollama Docker container."""
    _run_docker_lifecycle("start")


@server.command("stop")
def server_stop():
    """Stop the Ollama Docker container."""
    _run_docker_lifecycle("stop")


@server.command("restart")
def server_restart():
    """Restart the Ollama Docker container."""
    _run_docker_lifecycle("restart")


@server.command("serve")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=DEFAULT_API_PORT, type=int, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="Auto-reload on code changes (dev only).")
def server_serve(host, port, reload):
    """Start the GPU Directer FastAPI server (queue + /gd/* endpoints)."""
    _require_server_deps()
    from gpu_directer.server.api import run_server
    console.print(f"[bold]Starting GPU Directer API server on {host}:{port}…[/bold]")
    run_server(host=host, port=port, reload=reload)


def _run_docker_lifecycle(action: str):
    result = subprocess.run(
        ["docker", action, "ollama"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        console.print(f"[green]✓ Ollama container {action}ed successfully.[/green]")
    else:
        err_console.print(f"[red]Failed to {action} ollama container:[/red] {result.stderr.strip()}")
        sys.exit(2)


# ---------------------------------------------------------------------------
# client group
# ---------------------------------------------------------------------------

@main.group()
@click.option("--json", "json_output", is_flag=True, default=False)
@click.option("--quiet", is_flag=True, default=False)
@click.pass_context
def client(ctx, json_output, quiet):
    """Commands for the client machine (requires [client] extra)."""
    parent: _Ctx = ctx.obj
    if json_output:
        parent.json_output = True
    if quiet:
        parent.quiet = True


def _require_client_deps():
    from gpu_directer.client import HAVE_CLIENT
    if not HAVE_CLIENT:
        err_console.print(
            "[red]Client dependencies not installed. Install with:[/red]\n"
            "  pip install gpu-directer[client]"
        )
        sys.exit(1)


@client.command("setup")
@click.option("--server-ip", default=None)
@click.option("--port", default=DEFAULT_API_PORT, type=int, show_default=True)
@click.option("--non-interactive", is_flag=True, default=False)
@click.pass_context
def client_setup(ctx, server_ip, port, non_interactive):
    """Interactive wizard to connect a client to the GPU server."""
    _require_client_deps()
    from gpu_directer.client.setup_wizard import run_client_setup
    run_client_setup(
        server_ip=server_ip,
        port=port,
        non_interactive=non_interactive,
        config_path=ctx.obj.config_path,
    )


@client.command("status")
@click.option("--json", "json_flag", is_flag=True, default=False)
@click.pass_context
def client_status(ctx, json_flag):
    """Show current connection status between client and configured server."""
    _require_client_deps()
    from gpu_directer.client.status import get_client_status
    obj: _Ctx = ctx.obj
    status_data = get_client_status(config_path=obj.config_path)
    if obj.json_output or json_flag:
        click.echo(json.dumps(status_data, indent=2))
    else:
        _print_client_status(status_data)
    remote_ok = status_data.get("remote", {}).get("reachable", False)
    local_ok = status_data.get("local", {}).get("reachable", False)
    sys.exit(0 if (remote_ok or local_ok) else 1)


def _print_client_status(data):
    remote = data.get("remote", {})
    local = data.get("local", {})
    cfg = data.get("config", {})

    console.print("\n[bold]GPU Directer Client Status[/bold]")
    ip = cfg.get("server_ip", "not configured")
    port = cfg.get("server_port", DEFAULT_PORT)
    console.print(f"  Server IP:      {ip}:{port}")
    if remote.get("reachable"):
        console.print("  Server status:  [green]● online[/green]")
        console.print(f"  Queue depth:    {remote.get('queue_depth', 0)} requests waiting")
    else:
        console.print("  Server status:  [red]● offline[/red]")
    console.print(f"  Routing mode:   {cfg.get('routing_mode', 'auto')}")

    remote_models = remote.get("models", [])
    if remote_models:
        console.print(f"  Available models (remote):\n    {', '.join(remote_models)}")

    if local.get("reachable"):
        console.print("  Local Ollama:   [green]● online[/green]")
        local_models = local.get("models", [])
        if local_models:
            console.print(f"    Local models: {', '.join(local_models)}")
    else:
        console.print("  Local Ollama:   [red]● offline[/red]")


@client.command("models")
@click.option("--source", default="all", type=click.Choice(["remote", "local", "all"]))
@click.option("--json", "json_flag", is_flag=True, default=False)
@click.pass_context
def client_models(ctx, source, json_flag):
    """List models available on the remote server and/or local Ollama."""
    _require_client_deps()
    from gpu_directer.client.connectivity import query_server_models, query_local_models
    obj: _Ctx = ctx.obj
    from gpu_directer import config as cfg_mod
    config = cfg_mod.load_config(obj.config_path)
    server_ip = config.get("client", {}).get("server_ip", "")
    server_port = config.get("client", {}).get("server_port", DEFAULT_PORT)

    rows = []
    if source in ("remote", "all") and server_ip:
        remote_models = query_server_models(server_ip, server_port)
        for m in (remote_models or []):
            rows.append({"name": m, "source": "remote"})
    if source in ("local", "all"):
        local_models = query_local_models()
        for m in (local_models or []):
            rows.append({"name": m, "source": "local"})

    if obj.json_output or json_flag:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        console.print("No models found.")
        return
    from rich.table import Table
    table = Table(title="Available Models")
    table.add_column("Name")
    table.add_column("Source")
    for row in rows:
        table.add_row(row["name"], row["source"])
    console.print(table)


# ---------------------------------------------------------------------------
# config group
# ---------------------------------------------------------------------------

@main.group("config")
@click.option("--json", "json_output", is_flag=True, default=False)
@click.option("--quiet", is_flag=True, default=False)
@click.pass_context
def config_group(ctx, json_output, quiet):
    """Configuration management (available on all machines)."""
    parent: _Ctx = ctx.obj
    if json_output:
        parent.json_output = True
    if quiet:
        parent.quiet = True


@config_group.command("show")
@click.option("--json", "json_flag", is_flag=True, default=False)
@click.pass_context
def config_show(ctx, json_flag):
    """Print current configuration."""
    from gpu_directer import config as cfg_mod
    obj: _Ctx = ctx.obj
    data = cfg_mod.load_config(obj.config_path)
    if obj.json_output or json_flag:
        click.echo(json.dumps(data, indent=2))
        return
    from gpu_directer.core.constants import CONFIG_PATH
    path_display = obj.config_path or str(CONFIG_PATH)
    console.print(f"[dim]Config: {path_display}[/dim]\n")
    for section, values in data.items():
        console.print(f"[bold][{section}][/bold]")
        if isinstance(values, dict):
            for k, v in values.items():
                console.print(f"  {k} = {v!r}")
        console.print()


@config_group.command("set")
@click.argument("keyvalue")
@click.pass_context
def config_set(ctx, keyvalue):
    """Update a single config value (format: key=value)."""
    from gpu_directer import config as cfg_mod
    from gpu_directer.core.exceptions import GPUDirecterConfigError
    obj: _Ctx = ctx.obj
    if "=" not in keyvalue:
        err_console.print("[red]Error:[/red] Expected format: key=value (e.g. client.routing_mode=local)")
        sys.exit(1)
    key, _, value = keyvalue.partition("=")
    try:
        cfg_mod.set_value(key.strip(), value.strip(), obj.config_path)
        if not obj.quiet:
            console.print(f"[green]Set[/green] {key.strip()} = {value.strip()!r}")
    except GPUDirecterConfigError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)


@config_group.command("edit")
@click.pass_context
def config_edit(ctx):
    """Open config file in $EDITOR (falls back to nano)."""
    from gpu_directer import config as cfg_mod
    from gpu_directer.core.constants import CONFIG_PATH
    obj: _Ctx = ctx.obj
    config_path = Path(obj.config_path) if obj.config_path else CONFIG_PATH
    if not config_path.exists():
        cfg_mod.load_config(obj.config_path)  # creates default
    editor = os.environ.get("EDITOR", "nano")
    subprocess.run([editor, str(config_path)])


@config_group.command("reset")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.pass_context
def config_reset(ctx, yes):
    """Reset configuration to defaults."""
    from gpu_directer import config as cfg_mod
    obj: _Ctx = ctx.obj
    if not yes:
        click.confirm("Reset config to defaults? This will overwrite your current settings.", abort=True)
    cfg = cfg_mod.create_default_config()
    cfg_mod.save_config(cfg, obj.config_path)
    from gpu_directer.core.constants import CONFIG_PATH
    path_display = obj.config_path or str(CONFIG_PATH)
    console.print(f"[green]✓ Config reset to defaults:[/green] {path_display}")


# ---------------------------------------------------------------------------
# Missing import at top — needed for config edit
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402
