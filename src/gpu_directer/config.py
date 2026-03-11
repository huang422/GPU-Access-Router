"""TOML configuration read/write for GPU Directer."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from gpu_directer.core.constants import (
    CONFIG_PATH,
    DEFAULT_API_PORT,
    DEFAULT_PORT,
    DEFAULT_QUEUE_DEPTH,
    DEFAULT_ROUTING_MODE,
    DEFAULT_TIMEOUT,
)
from gpu_directer.core.exceptions import GPUDirecterConfigError

VALID_ROUTING_MODES = {"auto", "remote", "local"}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "client": {
        "server_ip": "",
        "server_port": DEFAULT_PORT,
        "routing_mode": DEFAULT_ROUTING_MODE,
        "timeout_seconds": DEFAULT_TIMEOUT,
        "default_model": "",
    },
    "server": {
        "ollama_port": DEFAULT_PORT,
        "api_port": DEFAULT_API_PORT,
        "queue_timeout": DEFAULT_TIMEOUT,
        "max_queue_depth": DEFAULT_QUEUE_DEPTH,
    },
    "meta": {
        "role": "client",
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
    },
}


def _resolve_path(path: Optional[str]) -> Path:
    return Path(path) if path else CONFIG_PATH


def create_default_config() -> Dict[str, Any]:
    """Return a fresh default configuration dict."""
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    cfg["meta"]["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return cfg


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from TOML file; create default if missing."""
    config_path = _resolve_path(path)
    if not config_path.exists():
        cfg = create_default_config()
        save_config(cfg, path)
        return cfg
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        return data
    except Exception as exc:
        raise GPUDirecterConfigError(f"Failed to read config at {config_path}: {exc}") from exc


def save_config(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Write config dict to TOML file."""
    config_path = _resolve_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)
    except Exception as exc:
        raise GPUDirecterConfigError(f"Failed to write config at {config_path}: {exc}") from exc


def get(key: str, default: Any = None, path: Optional[str] = None) -> Any:
    """Retrieve a value using dotted key access (e.g. 'client.server_ip')."""
    data = load_config(path)
    parts = key.split(".")
    node: Any = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def set_value(key: str, value: Any, path: Optional[str] = None) -> None:
    """Set a config value by dotted key (e.g. 'client.routing_mode')."""
    data = load_config(path)
    parts = key.split(".")
    if len(parts) < 2:
        raise GPUDirecterConfigError(f"Invalid config key '{key}'. Use section.field format.")

    node = data
    for part in parts[:-1]:
        if part not in node:
            raise GPUDirecterConfigError(f"Unknown config section '{part}'.")
        node = node[part]

    leaf = parts[-1]
    if leaf not in node:
        raise GPUDirecterConfigError(f"Unknown config key '{key}'.")

    coerced = _coerce(key, leaf, value, node[leaf])
    node[leaf] = coerced
    validate_config(data)
    save_config(data, path)


def _coerce(full_key: str, leaf: str, value: Any, existing: Any) -> Any:
    """Coerce string value from CLI to the correct type."""
    if isinstance(existing, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            raise GPUDirecterConfigError(f"'{full_key}' must be an integer, got '{value}'.")
    if isinstance(existing, bool):
        if str(value).lower() in ("true", "1", "yes"):
            return True
        if str(value).lower() in ("false", "0", "no"):
            return False
        raise GPUDirecterConfigError(f"'{full_key}' must be a boolean.")
    return str(value)


def validate_config(data: Dict[str, Any]) -> None:
    """Validate config structure; raise GPUDirecterConfigError on problems."""
    client = data.get("client", {})
    server = data.get("server", {})

    routing_mode = client.get("routing_mode", DEFAULT_ROUTING_MODE)
    if routing_mode not in VALID_ROUTING_MODES:
        raise GPUDirecterConfigError(
            f"Invalid routing_mode '{routing_mode}'. Must be one of: {', '.join(sorted(VALID_ROUTING_MODES))}."
        )

    for field in ("timeout_seconds",):
        val = client.get(field)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise GPUDirecterConfigError(f"client.{field} must be a positive integer.")

    for field in ("queue_timeout",):
        val = server.get(field)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise GPUDirecterConfigError(f"server.{field} must be a positive integer.")

    for field in ("server_port",):
        val = client.get(field)
        if val is not None and not (1 <= val <= 65535):
            raise GPUDirecterConfigError(f"client.{field} must be in range 1–65535.")

    for field in ("ollama_port", "api_port"):
        val = server.get(field)
        if val is not None and not (1 <= val <= 65535):
            raise GPUDirecterConfigError(f"server.{field} must be in range 1–65535.")
