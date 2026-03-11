"""GPURouter — primary developer-facing class."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from gpu_directer.core.constants import DEFAULT_API_PORT, DEFAULT_PORT, DEFAULT_ROUTING_MODE, DEFAULT_TIMEOUT
from gpu_directer.core.exceptions import GPUDirecterConfigError, GPUDirecterConnectionError


class GPURouter:
    """Route LLM inference to a remote GPU server or local Ollama.

    Usage::

        from gpu_directer import GPURouter

        router = GPURouter()
        response = router.chat("llama3.2", [{"role": "user", "content": "Hello!"}])
        print(response.message.content)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        routing_mode: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        from gpu_directer import config as cfg_mod
        from gpu_directer.core.exceptions import GPUDirecterConfigError

        try:
            self._config = cfg_mod.load_config(config_path)
            cfg_mod.validate_config(self._config)
        except GPUDirecterConfigError:
            raise
        except Exception as exc:
            raise GPUDirecterConfigError(f"Failed to load config: {exc}") from exc

        self._config_path = config_path
        client_cfg = self._config.get("client", {})

        self.routing_mode: str = routing_mode or client_cfg.get("routing_mode", DEFAULT_ROUTING_MODE)
        self.timeout: int = timeout or int(client_cfg.get("timeout_seconds", DEFAULT_TIMEOUT))
        self.server_ip: Optional[str] = client_cfg.get("server_ip") or None
        self.server_port: int = int(client_cfg.get("server_port", DEFAULT_API_PORT))

        self._remote_client = None
        self._local_client = None

    def _get_remote_client(self):
        if self._remote_client is None:
            import ollama
            self._remote_client = ollama.Client(
                host=f"http://{self.server_ip}:{self.server_port}"
            )
        return self._remote_client

    def _get_local_client(self):
        if self._local_client is None:
            import ollama
            self._local_client = ollama.Client(host=f"http://localhost:{DEFAULT_PORT}")
        return self._local_client

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        prefer: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        """Route and execute a chat inference call.

        Returns an ollama.ChatResponse (identical to ollama.Client.chat()).
        """
        from gpu_directer.client.routing import resolve_route
        from gpu_directer.client.poller import poll_for_result

        effective_timeout = timeout or self.timeout

        route = resolve_route(self._config, model, prefer=prefer or self.routing_mode)

        if route == "remote":
            return self._chat_remote(model, messages, effective_timeout, **kwargs)
        else:
            return self._chat_local(model, messages, **kwargs)

    def _chat_remote(self, model, messages, timeout, **kwargs):
        """POST to /gd/chat then poll for result."""
        from gpu_directer.client.poller import poll_for_result

        if not self.server_ip:
            raise GPUDirecterConnectionError("No server_ip configured.")

        base_url = f"http://{self.server_ip}:{self.server_port}"
        payload = {
            "model": model,
            "messages": messages,
            "options": kwargs.get("options", {}),
            "timeout": timeout,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/gd/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
        except Exception as exc:
            raise GPUDirecterConnectionError(
                f"Failed to submit request to {base_url}/gd/chat: {exc}"
            ) from exc

        request_id = result["request_id"]
        return poll_for_result(base_url, request_id, timeout)

    def _chat_local(self, model, messages, **kwargs):
        """Call local Ollama directly."""
        client = self._get_local_client()
        return client.chat(model=model, messages=messages, **kwargs)

    def list_models(self, source: str = "auto") -> Dict[str, Any]:
        """List models on remote, local, or both.

        Returns::

            {
                "remote": [...] or None,
                "local": [...] or None,
                "reachable": {"remote": bool, "local": bool}
            }
        """
        from gpu_directer.client.connectivity import (
            probe_server,
            query_local_models,
            query_server_models,
        )

        result: Dict[str, Any] = {
            "remote": None,
            "local": None,
            "reachable": {"remote": False, "local": False},
        }

        if source in ("remote", "auto") and self.server_ip:
            if probe_server(self.server_ip, self.server_port):
                result["reachable"]["remote"] = True
                models = query_server_models(self.server_ip, self.server_port)
                if models is not None:
                    result["remote"] = [{"name": m} for m in models]

        if source in ("local", "auto"):
            local_models = query_local_models()
            if local_models is not None:
                result["reachable"]["local"] = True
                result["local"] = [{"name": m} for m in local_models]

        return result

    def status(self) -> Dict[str, Any]:
        """Return connectivity and queue status for all sources."""
        from gpu_directer.client.connectivity import (
            probe_server,
            query_local_models,
            query_server_health,
            query_server_models,
        )
        from gpu_directer.core.constants import CONFIG_PATH

        remote: Dict[str, Any] = {
            "reachable": False,
            "server_ip": self.server_ip,
            "port": self.server_port,
        }
        if self.server_ip and probe_server(self.server_ip, self.server_port):
            remote["reachable"] = True
            health = query_server_health(self.server_ip, self.server_port) or {}
            remote["queue_depth"] = health.get("queue_depth", 0)
            remote["models"] = query_server_models(self.server_ip, self.server_port) or []

        local_models = query_local_models()
        local: Dict[str, Any] = {
            "reachable": local_models is not None,
            "models": local_models or [],
        }

        config_path_str = (
            self._config_path
            if self._config_path
            else str(CONFIG_PATH)
        )
        return {
            "remote": remote,
            "local": local,
            "routing_mode": self.routing_mode,
            "config_path": config_path_str,
        }
