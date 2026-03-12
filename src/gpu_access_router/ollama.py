"""Drop-in replacement for the ``ollama`` Python module.

Swap ``import ollama`` for ``from gpu_access_router import ollama`` and all
existing code continues to work — routing is controlled by config or env vars
instead of a hard-coded host.

Usage::

    from gpu_access_router import ollama

    # Identical to native ollama API:
    response = ollama.chat("llama3.2", [{"role": "user", "content": "Hello!"}])
    print(response.message.content)

    response = ollama.generate("llama3.2", "Tell me a joke")
    print(response.response)

    models = ollama.list()

Environment variables (override config file without editing it)::

    GPU_ROUTER_SERVER_IP       — remote GPU server IP
    GPU_ROUTER_ROUTING_MODE    — auto | local | remote
    GPU_ROUTER_FALLBACK_MODEL  — local model to use when remote fails
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gpu_access_router.client.router import GPURouter

# ---------------------------------------------------------------------------
# Singleton router — initialised on first use
# ---------------------------------------------------------------------------

_router: Optional[GPURouter] = None


def _get_router() -> GPURouter:
    global _router
    if _router is None:
        _router = GPURouter()
    return _router


# ---------------------------------------------------------------------------
# Module-level API (mirrors ollama.*)
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


def list(**kwargs: Any) -> Dict[str, Any]:
    """List models available on the configured source(s).

    Returns the same structure as ``GPURouter.list_models()``.
    """
    return _get_router().list_models(**kwargs)


# ---------------------------------------------------------------------------
# Client class (mirrors ollama.Client)
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

    def list(self, **kwargs: Any) -> Dict[str, Any]:
        return self._router.list_models(**kwargs)
