"""Routing decision logic for GPURouter."""

import warnings
from typing import Any, Dict, Optional

from gpu_directer.core.constants import DEFAULT_API_PORT
from gpu_directer.core.exceptions import GPUDirecterConnectionError


def resolve_route(
    config: Dict[str, Any],
    model: str,
    prefer: Optional[str] = None,
) -> str:
    """Return 'remote' or 'local' based on 4-step decision tree.

    Raises GPUDirecterConnectionError when no route is available.
    """
    from gpu_directer.client.connectivity import (
        probe_server,
        query_local_models,
        query_server_models,
    )

    client_cfg = config.get("client", {})
    server_ip = client_cfg.get("server_ip", "")
    server_port = int(client_cfg.get("server_port", DEFAULT_API_PORT))
    routing_mode = prefer or client_cfg.get("routing_mode", "auto")

    # Explicit remote-only
    if routing_mode == "remote":
        if not server_ip:
            raise GPUDirecterConnectionError(
                "routing_mode is 'remote' but no server_ip configured. "
                "Run: gpu-directer client setup"
            )
        if not probe_server(server_ip, server_port):
            raise GPUDirecterConnectionError(
                f"routing_mode is 'remote' but server {server_ip}:{server_port} is unreachable."
            )
        return "remote"

    # Explicit local-only
    if routing_mode == "local":
        local_models = query_local_models()
        if local_models is None:
            raise GPUDirecterConnectionError(
                "routing_mode is 'local' but local Ollama is not reachable at http://localhost:11434."
            )
        return "local"

    # Auto mode — 4-step decision tree
    # Step 1: Probe remote
    remote_reachable = bool(server_ip) and probe_server(server_ip, server_port)

    # Step 2 & 3: If remote reachable, check model availability
    if remote_reachable:
        remote_models = query_server_models(server_ip, server_port)

        # If model list is unavailable (query failed), route to remote optimistically —
        # the server will return an error if the model doesn't exist.
        if remote_models is None or model in remote_models:
            return "remote"

        # Step 4: Model not on remote — check local
        local_models = query_local_models()
        if local_models is not None and model in local_models:
            warnings.warn(
                f"Model '{model}' not found on remote server ({server_ip}:{server_port}) "
                f"but is available locally. Routing to local Ollama.",
                UserWarning,
                stacklevel=4,
            )
            return "local"

        # Model confirmed not on remote — show what IS available
        available = ", ".join(remote_models) if remote_models else "none"
        if local_models is None:
            raise GPUDirecterConnectionError(
                f"Model '{model}' not found on remote server and local Ollama is unreachable. "
                f"Remote has: [{available}]"
            )
        raise GPUDirecterConnectionError(
            f"Model '{model}' not found on remote server or local Ollama. "
            f"Remote has: [{available}]"
        )

    # Remote unreachable — fall back to local silently
    local_models = query_local_models()
    if local_models is not None:
        return "local"

    raise GPUDirecterConnectionError(
        "No routing target available. Remote server is unreachable and local Ollama is not running."
    )
