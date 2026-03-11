"""Client status helpers."""

from typing import Any, Dict, Optional

from gpu_directer.core.constants import DEFAULT_API_PORT


def get_client_status(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Query remote server health, remote models, and local Ollama."""
    from gpu_directer import config as cfg_mod
    from gpu_directer.client.connectivity import (
        probe_server,
        query_local_models,
        query_server_health,
        query_server_models,
    )

    cfg = cfg_mod.load_config(config_path)
    client_cfg = cfg.get("client", {})
    server_ip = client_cfg.get("server_ip", "")
    server_port = client_cfg.get("server_port", DEFAULT_API_PORT)
    routing_mode = client_cfg.get("routing_mode", "auto")

    # Remote status
    remote: Dict[str, Any] = {"reachable": False, "server_ip": server_ip, "port": server_port}
    if server_ip:
        if probe_server(server_ip, server_port, timeout=5):
            remote["reachable"] = True
            health = query_server_health(server_ip, server_port) or {}
            remote["queue_depth"] = health.get("queue_depth", 0)
            remote["models"] = query_server_models(server_ip, server_port) or []
        else:
            remote["models"] = []
    else:
        remote["models"] = []

    # Local status
    local_models = query_local_models()
    local: Dict[str, Any] = {
        "reachable": local_models is not None,
        "models": local_models or [],
    }

    return {
        "remote": remote,
        "local": local,
        "config": {
            "server_ip": server_ip,
            "server_port": server_port,
            "routing_mode": routing_mode,
        },
    }
