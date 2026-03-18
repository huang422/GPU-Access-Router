"""GPURouter — primary developer-facing class."""

import warnings
from typing import Any, Dict, Iterator, List, Optional

from gpu_access_router.core.constants import (
    DEFAULT_API_PORT,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PORT,
    DEFAULT_ROUTING_MODE,
    DEFAULT_TIMEOUT,
)
from gpu_access_router.core.exceptions import (
    GPUAccessRouterConfigError,
    GPUAccessRouterConnectionError,
    GPUAccessRouterTimeoutError,
)


class GPURouter:
    """Route LLM inference to a remote GPU server or local Ollama."""

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

        self.routing_mode: str = routing_mode or client_cfg.get(
            "routing_mode", DEFAULT_ROUTING_MODE
        )
        self.timeout: int = timeout or int(
            client_cfg.get("timeout_seconds", DEFAULT_TIMEOUT)
        )
        self.server_ip: Optional[str] = client_cfg.get("server_ip") or None
        self.server_port: int = int(
            client_cfg.get("server_port", DEFAULT_API_PORT)
        )
        self.fallback_model: str = client_cfg.get(
            "fallback_model", DEFAULT_FALLBACK_MODEL
        )

        self._local_clients: Dict[int, Any] = {}
        self._remote_clients: Dict[int, Any] = {}

    def _build_client(self, host: str, timeout_seconds: int):
        import httpx
        import ollama

        return ollama.Client(
            host=host,
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(timeout_seconds),
                write=30.0,
                pool=10.0,
            ),
        )

    def _get_local_client(self, timeout_seconds: Optional[int] = None):
        effective_timeout = timeout_seconds or self.timeout
        client = self._local_clients.get(effective_timeout)
        if client is None:
            client = self._build_client(
                host=f"http://localhost:{DEFAULT_PORT}",
                timeout_seconds=effective_timeout,
            )
            self._local_clients[effective_timeout] = client
        return client

    def _get_remote_client(self, timeout_seconds: Optional[int] = None):
        if not self.server_ip:
            raise GPUAccessRouterConnectionError("No server_ip configured.")

        effective_timeout = timeout_seconds or self.timeout
        client = self._remote_clients.get(effective_timeout)
        if client is None:
            client = self._build_client(
                host=f"http://{self.server_ip}:{self.server_port}",
                timeout_seconds=effective_timeout,
            )
            self._remote_clients[effective_timeout] = client
        return client

    def _local_target(self) -> str:
        return f"local Ollama (localhost:{DEFAULT_PORT})"

    def _remote_target(self) -> str:
        if not self.server_ip:
            return "remote GPU server"
        return f"remote GPU server ({self.server_ip}:{self.server_port})"

    @staticmethod
    def _is_streaming_call(kwargs: Dict[str, Any]) -> bool:
        return bool(kwargs.get("stream", False))

    def _map_client_exception(
        self,
        exc: Exception,
        *,
        target: str,
        timeout_seconds: int,
    ) -> Exception:
        import ollama

        try:
            import httpx
        except ImportError:  # pragma: no cover - client extra always provides httpx
            httpx = None  # type: ignore[assignment]

        if isinstance(
            exc,
            (GPUAccessRouterConnectionError, GPUAccessRouterTimeoutError),
        ):
            return exc

        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return GPUAccessRouterTimeoutError(
                f"Request to {target} timed out after {timeout_seconds}s."
            )

        if isinstance(exc, TimeoutError):
            return GPUAccessRouterTimeoutError(
                f"Request to {target} timed out after {timeout_seconds}s."
            )

        if isinstance(exc, ollama.ResponseError):
            error_text = getattr(exc, "error", str(exc))
            status_code = getattr(exc, "status_code", -1)
            lowered = error_text.lower()

            if status_code == 408 or "timed out" in lowered:
                return GPUAccessRouterTimeoutError(error_text)

            if status_code in (502, 503, 504) or any(
                marker in lowered
                for marker in ("failed to connect", "service unavailable", "queue full")
            ):
                return GPUAccessRouterConnectionError(error_text)

            return exc

        if isinstance(exc, ConnectionError):
            return GPUAccessRouterConnectionError(
                f"Failed to connect to {target}: {exc}"
            )

        return exc

    def _raise_mapped_client_exception(
        self,
        exc: Exception,
        *,
        target: str,
        timeout_seconds: int,
    ) -> None:
        mapped = self._map_client_exception(
            exc,
            target=target,
            timeout_seconds=timeout_seconds,
        )
        if mapped is exc:
            raise exc
        raise mapped from exc

    def _call_client_method(
        self,
        client: Any,
        method: str,
        *,
        target: str,
        timeout_seconds: int,
        **kwargs: Any,
    ):
        try:
            return getattr(client, method)(**kwargs)
        except Exception as exc:
            self._raise_mapped_client_exception(
                exc,
                target=target,
                timeout_seconds=timeout_seconds,
            )

    def _stream_client_method(
        self,
        client_getter,
        method: str,
        *,
        target: str,
        timeout_seconds: int,
        **kwargs: Any,
    ) -> Iterator[Any]:
        def iterator() -> Iterator[Any]:
            try:
                stream = getattr(client_getter(), method)(**kwargs)
                for item in stream:
                    yield item
            except Exception as exc:
                self._raise_mapped_client_exception(
                    exc,
                    target=target,
                    timeout_seconds=timeout_seconds,
                )

        return iterator()

    def _stream_with_optional_fallback(
        self,
        primary_stream_factory,
        fallback_stream_factory,
        *,
        should_fallback,
        warn_fallback,
    ) -> Iterator[Any]:
        def iterator() -> Iterator[Any]:
            yielded_any = False
            try:
                for item in primary_stream_factory():
                    yielded_any = True
                    yield item
            except GPUAccessRouterTimeoutError:
                raise
            except Exception as exc:
                if not yielded_any and should_fallback(exc):
                    warn_fallback()
                    for item in fallback_stream_factory():
                        yield item
                    return
                raise

        return iterator()

    @staticmethod
    def _warn_remote_fallback(model: str, fallback_model: str) -> None:
        warnings.warn(
            f"Remote inference failed for '{model}'. "
            f"Falling back to local model '{fallback_model}'.",
            UserWarning,
            stacklevel=2,
        )

    @staticmethod
    def _warn_local_fallback(model: str, fallback_model: str) -> None:
        warnings.warn(
            f"Local model '{model}' unavailable. "
            f"Trying fallback model '{fallback_model}'.",
            UserWarning,
            stacklevel=2,
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        prefer: Optional[str] = None,
        timeout: Optional[int] = None,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Route and execute a chat inference call."""
        from gpu_access_router.client.routing import resolve_route

        effective_timeout = timeout or self.timeout
        effective_fallback = fallback_model or self.fallback_model
        prefer_mode = prefer or self.routing_mode
        try:
            route = resolve_route(self._config, model, prefer=prefer_mode)
        except GPUAccessRouterConnectionError:
            if prefer_mode == "remote" and effective_fallback:
                self._warn_remote_fallback(model, effective_fallback)
                if self._is_streaming_call(kwargs):
                    return self._stream_client_method(
                        lambda: self._get_local_client(effective_timeout),
                        "chat",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        messages=messages,
                        **kwargs,
                    )
                return self._call_client_method(
                    self._get_local_client(effective_timeout),
                    "chat",
                    target=self._local_target(),
                    timeout_seconds=effective_timeout,
                    model=effective_fallback,
                    messages=messages,
                    **kwargs,
                )
            raise

        if route == "remote":
            if self._is_streaming_call(kwargs):
                primary_stream = lambda: self._stream_client_method(
                    lambda: self._get_remote_client(effective_timeout),
                    "chat",
                    target=self._remote_target(),
                    timeout_seconds=effective_timeout,
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                if effective_fallback:
                    return self._stream_with_optional_fallback(
                        primary_stream,
                        lambda: self._stream_client_method(
                            lambda: self._get_local_client(effective_timeout),
                            "chat",
                            target=self._local_target(),
                            timeout_seconds=effective_timeout,
                            model=effective_fallback,
                            messages=messages,
                            **kwargs,
                        ),
                        should_fallback=lambda exc: isinstance(
                            exc, GPUAccessRouterConnectionError
                        ),
                        warn_fallback=lambda: self._warn_remote_fallback(
                            model, effective_fallback
                        ),
                    )
                return primary_stream()

            try:
                return self._call_client_method(
                    self._get_remote_client(effective_timeout),
                    "chat",
                    target=self._remote_target(),
                    timeout_seconds=effective_timeout,
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            except GPUAccessRouterTimeoutError:
                raise
            except GPUAccessRouterConnectionError:
                if effective_fallback:
                    self._warn_remote_fallback(model, effective_fallback)
                    return self._call_client_method(
                        self._get_local_client(effective_timeout),
                        "chat",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        messages=messages,
                        **kwargs,
                    )
                raise

        if self._is_streaming_call(kwargs):
            primary_stream = lambda: self._stream_client_method(
                lambda: self._get_local_client(effective_timeout),
                "chat",
                target=self._local_target(),
                timeout_seconds=effective_timeout,
                model=model,
                messages=messages,
                **kwargs,
            )
            if effective_fallback and model != effective_fallback:
                return self._stream_with_optional_fallback(
                    primary_stream,
                    lambda: self._stream_client_method(
                        lambda: self._get_local_client(effective_timeout),
                        "chat",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        messages=messages,
                        **kwargs,
                    ),
                    should_fallback=lambda exc: not isinstance(
                        exc,
                        (
                            GPUAccessRouterConnectionError,
                            GPUAccessRouterTimeoutError,
                        ),
                    ),
                    warn_fallback=lambda: self._warn_local_fallback(
                        model, effective_fallback
                    ),
                )
            return primary_stream()

        try:
            return self._call_client_method(
                self._get_local_client(effective_timeout),
                "chat",
                target=self._local_target(),
                timeout_seconds=effective_timeout,
                model=model,
                messages=messages,
                **kwargs,
            )
        except (GPUAccessRouterConnectionError, GPUAccessRouterTimeoutError):
            raise
        except Exception:
            if effective_fallback and model != effective_fallback:
                self._warn_local_fallback(model, effective_fallback)
                return self._call_client_method(
                    self._get_local_client(effective_timeout),
                    "chat",
                    target=self._local_target(),
                    timeout_seconds=effective_timeout,
                    model=effective_fallback,
                    messages=messages,
                    **kwargs,
                )
            raise

    def generate(
        self,
        model: str,
        prompt: str,
        prefer: Optional[str] = None,
        timeout: Optional[int] = None,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Route and execute a generate inference call."""
        from gpu_access_router.client.routing import resolve_route

        effective_timeout = timeout or self.timeout
        effective_fallback = fallback_model or self.fallback_model
        prefer_mode = prefer or self.routing_mode
        try:
            route = resolve_route(self._config, model, prefer=prefer_mode)
        except GPUAccessRouterConnectionError:
            if prefer_mode == "remote" and effective_fallback:
                self._warn_remote_fallback(model, effective_fallback)
                if self._is_streaming_call(kwargs):
                    return self._stream_client_method(
                        lambda: self._get_local_client(effective_timeout),
                        "generate",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        prompt=prompt,
                        **kwargs,
                    )
                return self._call_client_method(
                    self._get_local_client(effective_timeout),
                    "generate",
                    target=self._local_target(),
                    timeout_seconds=effective_timeout,
                    model=effective_fallback,
                    prompt=prompt,
                    **kwargs,
                )
            raise

        if route == "remote":
            if self._is_streaming_call(kwargs):
                primary_stream = lambda: self._stream_client_method(
                    lambda: self._get_remote_client(effective_timeout),
                    "generate",
                    target=self._remote_target(),
                    timeout_seconds=effective_timeout,
                    model=model,
                    prompt=prompt,
                    **kwargs,
                )
                if effective_fallback:
                    return self._stream_with_optional_fallback(
                        primary_stream,
                        lambda: self._stream_client_method(
                            lambda: self._get_local_client(effective_timeout),
                            "generate",
                            target=self._local_target(),
                            timeout_seconds=effective_timeout,
                            model=effective_fallback,
                            prompt=prompt,
                            **kwargs,
                        ),
                        should_fallback=lambda exc: isinstance(
                            exc, GPUAccessRouterConnectionError
                        ),
                        warn_fallback=lambda: self._warn_remote_fallback(
                            model, effective_fallback
                        ),
                    )
                return primary_stream()

            try:
                return self._call_client_method(
                    self._get_remote_client(effective_timeout),
                    "generate",
                    target=self._remote_target(),
                    timeout_seconds=effective_timeout,
                    model=model,
                    prompt=prompt,
                    **kwargs,
                )
            except GPUAccessRouterTimeoutError:
                raise
            except GPUAccessRouterConnectionError:
                if effective_fallback:
                    self._warn_remote_fallback(model, effective_fallback)
                    return self._call_client_method(
                        self._get_local_client(effective_timeout),
                        "generate",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        prompt=prompt,
                        **kwargs,
                    )
                raise

        if self._is_streaming_call(kwargs):
            primary_stream = lambda: self._stream_client_method(
                lambda: self._get_local_client(effective_timeout),
                "generate",
                target=self._local_target(),
                timeout_seconds=effective_timeout,
                model=model,
                prompt=prompt,
                **kwargs,
            )
            if effective_fallback and model != effective_fallback:
                return self._stream_with_optional_fallback(
                    primary_stream,
                    lambda: self._stream_client_method(
                        lambda: self._get_local_client(effective_timeout),
                        "generate",
                        target=self._local_target(),
                        timeout_seconds=effective_timeout,
                        model=effective_fallback,
                        prompt=prompt,
                        **kwargs,
                    ),
                    should_fallback=lambda exc: not isinstance(
                        exc,
                        (
                            GPUAccessRouterConnectionError,
                            GPUAccessRouterTimeoutError,
                        ),
                    ),
                    warn_fallback=lambda: self._warn_local_fallback(
                        model, effective_fallback
                    ),
                )
            return primary_stream()

        try:
            return self._call_client_method(
                self._get_local_client(effective_timeout),
                "generate",
                target=self._local_target(),
                timeout_seconds=effective_timeout,
                model=model,
                prompt=prompt,
                **kwargs,
            )
        except (GPUAccessRouterConnectionError, GPUAccessRouterTimeoutError):
            raise
        except Exception:
            if effective_fallback and model != effective_fallback:
                self._warn_local_fallback(model, effective_fallback)
                return self._call_client_method(
                    self._get_local_client(effective_timeout),
                    "generate",
                    target=self._local_target(),
                    timeout_seconds=effective_timeout,
                    model=effective_fallback,
                    prompt=prompt,
                    **kwargs,
                )
            raise

    def list(self, prefer: Optional[str] = None):
        """List models from the routed source using the native Ollama response type."""
        from gpu_access_router.client.routing import resolve_list_route

        route = resolve_list_route(self._config, prefer=prefer or self.routing_mode)
        if route == "remote":
            return self._call_client_method(
                self._get_remote_client(self.timeout),
                "list",
                target=self._remote_target(),
                timeout_seconds=self.timeout,
            )
        return self._call_client_method(
            self._get_local_client(self.timeout),
            "list",
            target=self._local_target(),
            timeout_seconds=self.timeout,
        )

    def list_models(self, source: str = "auto") -> Dict[str, Any]:
        """List models on remote, local, or both."""
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

        config_path_str = self._config_path if self._config_path else str(CONFIG_PATH)
        return {
            "remote": remote,
            "local": local,
            "routing_mode": self.routing_mode,
            "config_path": config_path_str,
        }
