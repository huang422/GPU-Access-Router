"""Drop-in replacement for the ``ollama`` Python module.

Swap ``import ollama`` for ``from gpu_access_router import ollama`` and all
existing code continues to work — routing is controlled by config or env vars
instead of a hard-coded host.

Supports both **sync** and **async** usage::

    # Sync (drop-in for ollama.Client):
    from gpu_access_router.ollama import Client
    response = Client().generate("qwen3.5:9b", "Hello")

    # Async (drop-in for ollama.AsyncClient):
    from gpu_access_router.ollama import AsyncClient
    async with AsyncClient() as client:
        response = await client.generate("qwen3.5:9b", "Hello", stream=True)
        async for chunk in response:
            print(chunk.response, end="")

Environment variables (override config file without editing it)::

    GPU_ROUTER_SERVER_IP       — remote GPU server IP
    GPU_ROUTER_ROUTING_MODE    — auto | local | remote
    GPU_ROUTER_FALLBACK_MODEL  — local model to use when remote fails
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import ollama as _ollama

from gpu_access_router.client.router import GPURouter
from gpu_access_router.core.constants import DEFAULT_PORT
from gpu_access_router.core.exceptions import GPUAccessRouterConnectionError

logger = logging.getLogger("gpu_access_router.ollama")

# ---------------------------------------------------------------------------
# Singleton routers — initialised on first use
# ---------------------------------------------------------------------------

_router: Optional[GPURouter] = None
_async_client: Optional["AsyncClient"] = None


def _get_router() -> GPURouter:
    global _router
    if _router is None:
        _router = GPURouter()
    return _router


def _get_async_client() -> "AsyncClient":
    global _async_client
    if _async_client is None:
        _async_client = AsyncClient()
    return _async_client


# ---------------------------------------------------------------------------
# Connection error detection
# ---------------------------------------------------------------------------

def _is_connection_error(exc: Exception) -> bool:
    """Check if an exception indicates a connection/network failure."""
    connection_types = [ConnectionError, TimeoutError, OSError, GPUAccessRouterConnectionError]
    try:
        import httpx
        connection_types.extend([httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout])
    except ImportError:
        pass
    connection_types = tuple(connection_types)
    return isinstance(exc, connection_types)


# ---------------------------------------------------------------------------
# Module-level sync API (mirrors ollama.*)
# ---------------------------------------------------------------------------

def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    fallback_model: Optional[str] = None,
    **kwargs: Any,
):
    """Send a chat request, routing to remote or local Ollama as configured.

    Returns an ``ollama.ChatResponse``.
    """
    return _get_router().chat(model=model, messages=messages, fallback_model=fallback_model, **kwargs)


def generate(model: str, prompt: str, *, fallback_model: Optional[str] = None, **kwargs: Any):
    """Generate a completion, routing to remote or local Ollama as configured.

    Returns an ``ollama.GenerateResponse``.
    """
    return _get_router().generate(model=model, prompt=prompt, fallback_model=fallback_model, **kwargs)


def list():
    """List models from the routed source.

    Returns the native ``ollama.ListResponse`` type.
    """
    return _get_router().list()


# ---------------------------------------------------------------------------
# Module-level async API
# ---------------------------------------------------------------------------

async def agenerate(model: str, prompt: str, *, fallback_model: Optional[str] = None, **kwargs: Any):
    """Async generate with automatic routing and fallback."""
    return await _get_async_client().generate(model=model, prompt=prompt, fallback_model=fallback_model, **kwargs)


async def achat(model: str, messages: List[Dict[str, Any]], *, fallback_model: Optional[str] = None, **kwargs: Any):
    """Async chat with automatic routing and fallback."""
    return await _get_async_client().chat(model=model, messages=messages, fallback_model=fallback_model, **kwargs)


async def alist(**kwargs: Any):
    """Async list models from the routed source."""
    return await _get_async_client().list(**kwargs)


# ---------------------------------------------------------------------------
# Client class (sync, mirrors ollama.Client)
# ---------------------------------------------------------------------------

class Client:
    """Drop-in replacement for ``ollama.Client``.

    Instead of accepting a ``host`` URL, accepts ``routing_mode`` and
    ``fallback_model`` to control where inference runs.

    Example::

        from gpu_access_router.ollama import Client
        client = Client(routing_mode="auto", fallback_model="qwen3.5:9b")
        response = client.chat("qwen3.5:35b", [...])
    """

    def __init__(
        self,
        host: Optional[str] = None,  # accepted but ignored; routing is via config
        routing_mode: Optional[str] = None,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._router = GPURouter(routing_mode=routing_mode)
        self._fallback_model = fallback_model

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ):
        return self._router.chat(
            model=model,
            messages=messages,
            fallback_model=fallback_model or self._fallback_model,
            **kwargs,
        )

    def generate(self, model: str, prompt: str, *, fallback_model: Optional[str] = None, **kwargs: Any):
        return self._router.generate(
            model=model,
            prompt=prompt,
            fallback_model=fallback_model or self._fallback_model,
            **kwargs,
        )

    def list(self):
        return self._router.list(prefer=self._router.routing_mode)


# ---------------------------------------------------------------------------
# AsyncClient class (async, mirrors ollama.AsyncClient)
# ---------------------------------------------------------------------------

class AsyncClient:
    """Async drop-in replacement for ``ollama.AsyncClient``.

    Routes inference to remote GPU-Access-Router server or local Ollama,
    with automatic fallback on connection failure. All Ollama parameters
    (``think``, ``images``, ``stream``, ``system``, ``options``, etc.)
    pass through transparently.

    Example::

        from gpu_access_router.ollama import AsyncClient

        async with AsyncClient() as client:
            # Streaming
            response = await client.generate("qwen3.5:9b", "Hello", stream=True)
            async for chunk in response:
                print(chunk.response, end="")

            # Non-streaming
            resp = await client.generate("qwen3.5:9b", "Hello")
            print(resp.response)

            # Think mode + multimodal (all params pass through)
            resp = await client.generate(
                "qwen3.5:35b-a3b", "Describe this image",
                think=True, images=[base64_data],
                options={"num_predict": 6144, "temperature": 0.7},
            )
    """

    def __init__(
        self,
        host: Optional[str] = None,  # accepted but ignored; routing is via config
        routing_mode: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._gpu_router = GPURouter(routing_mode=routing_mode)
        self._fallback_model = fallback_model or self._gpu_router.fallback_model
        self._timeout = timeout or self._gpu_router.timeout
        self._remote_client: Optional[_ollama.AsyncClient] = None
        self._local_client: Optional[_ollama.AsyncClient] = None

    def _get_remote_client(self) -> _ollama.AsyncClient:
        """Lazy-init remote client pointing to GPU-Access-Router server."""
        if self._remote_client is None:
            import httpx
            ip = self._gpu_router.server_ip
            port = self._gpu_router.server_port
            if not ip:
                raise GPUAccessRouterConnectionError("No server_ip configured for remote routing.")
            self._remote_client = _ollama.AsyncClient(
                host=f"http://{ip}:{port}",
                timeout=httpx.Timeout(connect=10.0, read=float(self._timeout), write=30.0, pool=10.0),
            )
        return self._remote_client

    def _get_local_client(self) -> _ollama.AsyncClient:
        """Lazy-init local client pointing to localhost Ollama."""
        if self._local_client is None:
            import httpx
            self._local_client = _ollama.AsyncClient(
                host=f"http://localhost:{DEFAULT_PORT}",
                timeout=httpx.Timeout(connect=10.0, read=float(self._timeout), write=30.0, pool=10.0),
            )
        return self._local_client

    async def generate(
        self,
        model: str = "",
        prompt: Optional[str] = None,
        *,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Generate with automatic routing and fallback.

        Accepts all ``ollama.AsyncClient.generate()`` parameters:
        ``stream``, ``think``, ``system``, ``images``, ``options``, etc.
        """
        return await self._route_call("generate", model, fallback_model, prompt=prompt, **kwargs)

    async def chat(
        self,
        model: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        fallback_model: Optional[str] = None,
        **kwargs: Any,
    ):
        """Chat with automatic routing and fallback.

        Accepts all ``ollama.AsyncClient.chat()`` parameters:
        ``stream``, ``think``, ``tools``, ``options``, etc.
        """
        return await self._route_call("chat", model, fallback_model, messages=messages, **kwargs)

    async def list(self, **kwargs: Any):
        """List models from the routed source."""
        try:
            route = self._resolve_route("")
        except GPUAccessRouterConnectionError:
            route = "local"
        client = self._get_remote_client() if route == "remote" else self._get_local_client()
        return await client.list(**kwargs)

    def _resolve_route(self, model: str) -> str:
        """Resolve routing for a model, returning 'remote' or 'local'."""
        from gpu_access_router.client.routing import resolve_route
        return resolve_route(self._gpu_router._config, model, prefer=self._gpu_router.routing_mode)

    async def _route_call(self, method: str, model: str, fallback_model: Optional[str], **kwargs: Any):
        """Core routing: try primary route, fallback on connection error."""
        effective_fallback = fallback_model or self._fallback_model
        route = self._resolve_route(model)

        if route == "remote":
            try:
                client = self._get_remote_client()
                return await getattr(client, method)(model=model, **kwargs)
            except Exception as exc:
                if effective_fallback and _is_connection_error(exc):
                    logger.warning(
                        "Remote inference failed for '%s': %s. Falling back to local '%s'.",
                        model, exc, effective_fallback,
                    )
                    warnings.warn(
                        f"Remote inference failed for '{model}'. "
                        f"Falling back to local model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    client = self._get_local_client()
                    return await getattr(client, method)(model=effective_fallback, **kwargs)
                raise
        else:
            try:
                client = self._get_local_client()
                return await getattr(client, method)(model=model, **kwargs)
            except Exception as exc:
                if effective_fallback and model != effective_fallback and _is_connection_error(exc):
                    logger.warning(
                        "Local model '%s' failed: %s. Trying fallback '%s'.",
                        model, exc, effective_fallback,
                    )
                    warnings.warn(
                        f"Local model '{model}' unavailable. "
                        f"Trying fallback model '{effective_fallback}'.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return await getattr(client, method)(model=effective_fallback, **kwargs)
                raise

    async def close(self) -> None:
        """Close underlying HTTP clients."""
        if self._remote_client is not None:
            try:
                await self._remote_client._client.aclose()
            except Exception:
                pass
        if self._local_client is not None:
            try:
                await self._local_client._client.aclose()
            except Exception:
                pass

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
