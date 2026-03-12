"""TOML configuration read/write for GPU Access Router."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

from gpu_access_router.core.constants import (
    CONFIG_PATH,
    DEFAULT_API_PORT,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PORT,
    DEFAULT_QUEUE_DEPTH,
    DEFAULT_ROUTING_MODE,
    DEFAULT_TIMEOUT,
)
from gpu_access_router.core.exceptions import GPUAccessRouterConfigError

VALID_ROUTING_MODES = {"auto", "remote", "local"}

# Environment variables that override individual config fields.
# Maps env var name → (section, key).
_ENV_OVERRIDES = {
    "GPU_ROUTER_SERVER_IP":      ("client", "server_ip"),
    "GPU_ROUTER_ROUTING_MODE":   ("client", "routing_mode"),
    "GPU_ROUTER_FALLBACK_MODEL": ("client", "fallback_model"),
}

_DEFAULT_CONFIG: Dict[str, Any] = {
    "client": {
        "server_ip": "",
        "server_port": DEFAULT_API_PORT,
        "routing_mode": DEFAULT_ROUTING_MODE,
        "timeout_seconds": DEFAULT_TIMEOUT,
        "default_model": "",
        "fallback_model": DEFAULT_FALLBACK_MODEL,
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
    if path:
        return Path(path)
    env_path = os.environ.get("GPU_ACCESS_ROUTER_CONFIG")
    if env_path:
        return Path(env_path)
    return CONFIG_PATH


def create_default_config() -> Dict[str, Any]:
    """Return a fresh default configuration dict."""
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    cfg["meta"]["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return cfg


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load config from TOML file; create default if missing.

    Environment variables override individual fields after the file is loaded:
      GPU_ROUTER_SERVER_IP       → client.server_ip
      GPU_ROUTER_ROUTING_MODE    → client.routing_mode
      GPU_ROUTER_FALLBACK_MODEL  → client.fallback_model
    """
    config_path = _resolve_path(path)
    if not config_path.exists():
        cfg = create_default_config()
        save_config(cfg, path)
    else:
        try:
            with open(config_path, "rb") as f:
                cfg = tomllib.load(f)
        except Exception as exc:
            raise GPUAccessRouterConfigError(f"Failed to read config at {config_path}: {exc}") from exc

    # Apply environment variable overrides (do not persist to file).
    for env_var, (section, key) in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            cfg.setdefault(section, {})[key] = value

    return cfg


def save_config(data: Dict[str, Any], path: Optional[str] = None) -> None:
    """Write config dict to TOML file."""
    config_path = _resolve_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)
    except Exception as exc:
        raise GPUAccessRouterConfigError(f"Failed to write config at {config_path}: {exc}") from exc


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
    """Set a config value by dotted key (e.g. 'client.routing_mode').

    Validates the key against the default config schema so that new fields
    added to defaults (e.g. ``client.fallback_model``) are always settable,
    even if an existing config file pre-dates the field.
    """
    data = load_config(path)
    parts = key.split(".")
    if len(parts) < 2:
        raise GPUAccessRouterConfigError(f"Invalid config key '{key}'. Use section.field format.")

    section, leaf = parts[0], parts[-1]

    # Use the default config as the authoritative schema.
    default_section = _DEFAULT_CONFIG.get(section)
    if default_section is None:
        raise GPUAccessRouterConfigError(f"Unknown config section '{section}'.")
    if leaf not in default_section:
        raise GPUAccessRouterConfigError(f"Unknown config key '{key}'.")

    # Ensure the section exists in the loaded config.
    data.setdefault(section, {})
    existing = data[section].get(leaf, default_section[leaf])

    coerced = _coerce(key, leaf, value, existing)
    data[section][leaf] = coerced
    validate_config(data)
    save_config(data, path)


def _coerce(full_key: str, leaf: str, value: Any, existing: Any) -> Any:
    """Coerce string value from CLI to the correct type."""
    # bool must be checked before int because bool is a subclass of int in Python.
    if isinstance(existing, bool):
        if str(value).lower() in ("true", "1", "yes"):
            return True
        if str(value).lower() in ("false", "0", "no"):
            return False
        raise GPUAccessRouterConfigError(f"'{full_key}' must be a boolean.")
    if isinstance(existing, int):
        try:
            return int(value)
        except (ValueError, TypeError):
            raise GPUAccessRouterConfigError(f"'{full_key}' must be an integer, got '{value}'.")
    return str(value)


def validate_config(data: Dict[str, Any]) -> None:
    """Validate config structure; raise GPUAccessRouterConfigError on problems."""
    client = data.get("client", {})
    server = data.get("server", {})

    routing_mode = client.get("routing_mode", DEFAULT_ROUTING_MODE)
    if routing_mode not in VALID_ROUTING_MODES:
        raise GPUAccessRouterConfigError(
            f"Invalid routing_mode '{routing_mode}'. Must be one of: {', '.join(sorted(VALID_ROUTING_MODES))}."
        )

    for field in ("timeout_seconds",):
        val = client.get(field)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise GPUAccessRouterConfigError(f"client.{field} must be a positive integer.")

    for field in ("queue_timeout",):
        val = server.get(field)
        if val is not None and (not isinstance(val, int) or val <= 0):
            raise GPUAccessRouterConfigError(f"server.{field} must be a positive integer.")

    for field in ("server_port",):
        val = client.get(field)
        if val is not None and not (1 <= val <= 65535):
            raise GPUAccessRouterConfigError(f"client.{field} must be in range 1–65535.")

    for field in ("ollama_port", "api_port"):
        val = server.get(field)
        if val is not None and not (1 <= val <= 65535):
            raise GPUAccessRouterConfigError(f"server.{field} must be in range 1–65535.")
