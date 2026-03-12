"""GPURouter — primary developer-facing class."""

import json
import urllib.error
import urllib.request
import warnings
from typing import Any, Dict, List, Optional

from gpu_access_router.core.constants import DEFAULT_API_PORT, DEFAULT_FALLBACK_MODEL, DEFAULT_PORT, DEFAULT_ROUTING_MODE, DEFAULT_TIMEOUT
from gpu_access_router.core.exceptions import GPUAccessRouterConfigError, GPUAccessRouterConnectionError


class _GenerateResponse:
    """Minimal wrapper so remote generate() responses have a ``.response`` attribute."""

    def __init__(self, text: str) -> None:
        self.response = text

    def __repr__(self) -> str:  # pragma: no cover
        return f"_GenerateResponse(response={self.response!r})"


class GPURouter:
    """Route LLM inference to a remote GPU server or local Ollama.

    Usage::

        from gpu_access_router import GPURouter

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
        from gpu_access_router import config as cfg_mod

        try:
            self._config = cfg_mod.load_config(config_path)
            cfg_mod.validate_config(self._config)
        except GPUAccessRouterConfigError:
            raise
        except Exception as exc:
            raise GPUAccessRouterConfigError(f"Failed to load config: {exc}") from exc

        self._config_path = config_path
        client_cfg = self._config.get("client", {})

        self.routing_mode: str = routing_mode or client_cfg.get("routing_mode", DEFAULT_ROUTING_MODE)
        self.timeout: int = timeout or int(client_cfg.get("timeout_seconds", DEFAULT_TIMEOUT))
        self.server_ip: Optional[str] = client_cfg.get("server_ip") or None
        self.server_port: int = int(client_cfg.get("server_port", DEFAULT_API_PORT))
        self.fallback_model: str = client_cfg.get("fallback_model", DEFAULT_FALLBACK_MODEL)

        self._local_client = None

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
        fallback_model: Optional[str] = None,
        **kwargs,
    ):
        """Route and execute a chat inference call.

        Returns an ollama.ChatResponse (identical to ollama.Client.chat()).

        If ``fallback_model`` is provided (or configured via ``client.fallback_model``
        in config / ``GPU_ROUTER_FALLBACK_MODEL`` env var), it is used as a local
        fallback when the primary route fails (e.g. remote inference error, or the
        requested model is not available locally).
        """
        from gpu_access_router.client.routing import resolve_route

        effective_timeout = timeout or self.timeout
        effective_fallback = fallback_model or self.fallback_model

        route = resolve_route(self._config, model, prefer=prefer or self.routing_mode)

        if route == "remote":
            try:
                return self._chat_remote(model, messages, effective_timeout, **kwargs)
            except GPUAccessRouterConnectionError:
                if effective_fallback:
                    warnings.warn(
                        f"Remote inference failed for '{model}'. "
                        f"Falling back to local model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return self._chat_local(effective_fallback, messages, **kwargs)
                raise
        else:
            try:
                return self._chat_local(model, messages, **kwargs)
            except Exception:
                if effective_fallback and model != effective_fallback:
                    warnings.warn(
                        f"Local model '{model}' unavailable. "
                        f"Trying fallback model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return self._chat_local(effective_fallback, messages, **kwargs)
                raise

    def _chat_remote(self, model, messages, timeout, **kwargs):
        """POST to /gd/chat and wait for the result (server blocks until inference completes)."""
        from gpu_access_router.client.poller import _reconstruct_chat_response

        if not self.server_ip:
            raise GPUAccessRouterConnectionError("No server_ip configured.")

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
        # Socket timeout must exceed inference timeout so we receive the server's response
        http_timeout = timeout + 30
        try:
            with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 503:
                raise GPUAccessRouterConnectionError(
                    f"Server queue full or unavailable. "
                    f"Run 'gpu-access-router server restart' on the server. Detail: {body}"
                ) from exc
            if exc.code == 408:
                raise GPUAccessRouterConnectionError(
                    f"Inference timed out on server after {timeout}s."
                ) from exc
            raise GPUAccessRouterConnectionError(
                f"Server returned HTTP {exc.code}: {body}"
            ) from exc
        except Exception as exc:
            raise GPUAccessRouterConnectionError(
                f"Failed to connect to {base_url}/gd/chat: {exc}"
            ) from exc

        return _reconstruct_chat_response(result)

    def _chat_local(self, model, messages, **kwargs):
        """Call local Ollama directly."""
        client = self._get_local_client()
        return client.chat(model=model, messages=messages, **kwargs)

    def generate(
        self,
        model: str,
        prompt: str,
        prefer: Optional[str] = None,
        timeout: Optional[int] = None,
        fallback_model: Optional[str] = None,
        **kwargs,
    ):
        """Route and execute a generate inference call.

        For local routes uses ``ollama.Client.generate()`` natively.
        For remote routes wraps the prompt as a chat message and calls
        ``/gd/chat``, returning a response with a ``.response`` attribute.
        """
        from gpu_access_router.client.routing import resolve_route

        effective_timeout = timeout or self.timeout
        effective_fallback = fallback_model or self.fallback_model

        route = resolve_route(self._config, model, prefer=prefer or self.routing_mode)

        if route == "remote":
            try:
                return self._generate_remote(model, prompt, effective_timeout, **kwargs)
            except GPUAccessRouterConnectionError:
                if effective_fallback:
                    warnings.warn(
                        f"Remote inference failed for '{model}'. "
                        f"Falling back to local model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return self._generate_local(effective_fallback, prompt, **kwargs)
                raise
        else:
            try:
                return self._generate_local(model, prompt, **kwargs)
            except Exception:
                if effective_fallback and model != effective_fallback:
                    warnings.warn(
                        f"Local model '{model}' unavailable. "
                        f"Trying fallback model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return self._generate_local(effective_fallback, prompt, **kwargs)
                raise

    def _generate_remote(self, model: str, prompt: str, timeout: int, **kwargs):
        """Send generate as a chat message to /gd/chat; wrap response."""
        messages = [{"role": "user", "content": prompt}]
        chat_response = self._chat_remote(model, messages, timeout, **kwargs)
        return _GenerateResponse(chat_response.message.content)

    def _generate_local(self, model: str, prompt: str, **kwargs):
        """Call local Ollama generate directly."""
        client = self._get_local_client()
        return client.generate(model=model, prompt=prompt, **kwargs)

    def list_models(self, source: str = "auto") -> Dict[str, Any]:
        """List models on remote, local, or both.

        Returns::

            {
                "remote": [...] or None,
                "local": [...] or None,
                "reachable": {"remote": bool, "local": bool}
            }
        """
        from gpu_access_router.client.connectivity import (
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
        from gpu_access_router.client.connectivity import (
            probe_server,
            query_local_models,
            query_server_health,
            query_server_models,
        )
        from gpu_access_router.core.constants import CONFIG_PATH

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
