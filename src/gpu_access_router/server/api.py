"""FastAPI server exposing /gd/* endpoints and Ollama-compatible /api/* proxy with serial queue."""

import asyncio
import json
import logging
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    import uvicorn
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-access-router[server]"
    ) from exc

try:
    import ollama as _ollama
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-access-router[server]"
    ) from exc

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

from gpu_access_router.core.constants import DEFAULT_API_PORT, DEFAULT_PORT, DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT
from gpu_access_router.server.queue import SerialQueue

logger = logging.getLogger("gpu_access_router.server")

_serial_queue: Optional[SerialQueue] = None
_ollama_port: int = DEFAULT_PORT
_httpx_client: Optional["httpx.AsyncClient"] = None


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    options: Dict[str, Any] = {}
    timeout: int = DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _serial_queue, _ollama_port, _httpx_client
    from gpu_access_router import config as cfg_mod
    cfg = cfg_mod.load_config()
    _ollama_port = cfg.get("server", {}).get("ollama_port", DEFAULT_PORT)
    queue_timeout = cfg.get("server", {}).get("queue_timeout", DEFAULT_TIMEOUT)
    max_depth = cfg.get("server", {}).get("max_queue_depth", DEFAULT_QUEUE_DEPTH)

    _serial_queue = SerialQueue(timeout_seconds=queue_timeout, max_depth=max_depth)

    ollama_client = _ollama.Client(host=f"http://localhost:{_ollama_port}")

    def _ollama_call(model, messages, options):
        resp = ollama_client.chat(
            model=model,
            messages=messages,
            options=options or {},
            keep_alive=0,  # release GPU memory immediately after response
        )
        if hasattr(resp, "__dict__"):
            return _chat_response_to_dict(resp)
        return resp

    def _unload_model(model: str) -> None:
        try:
            ollama_client.generate(model=model, prompt="", keep_alive=0)
        except Exception:
            pass

    _serial_queue.set_ollama_callable(_ollama_call)
    _serial_queue.set_unload_callable(_unload_model)
    _serial_queue.start()

    # Initialize httpx client for Ollama-compatible proxy endpoints
    if httpx is not None:
        _httpx_client = httpx.AsyncClient(
            base_url=f"http://localhost:{_ollama_port}",
            timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
        )

    yield

    # Cleanup
    if _httpx_client is not None:
        await _httpx_client.aclose()


app = FastAPI(title="GPU Access Router Server", version="0.1.0", lifespan=lifespan)


def _chat_response_to_dict(resp) -> Dict:
    msg = resp.message
    return {
        "model": getattr(resp, "model", ""),
        "created_at": str(getattr(resp, "created_at", "")),
        "message": {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", ""),
        },
        "done": getattr(resp, "done", True),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/gd/health")
async def health():
    queue_depth = await _serial_queue.get_depth() if _serial_queue else 0
    has_gd_processing = any(
        r.status == "processing"
        for r in (_serial_queue._pending.values() if _serial_queue else [])
    )
    has_proxy_active = bool(_serial_queue._active_slots) if _serial_queue else False
    processing = has_gd_processing or has_proxy_active
    ollama_ok = _check_ollama_reachable(_ollama_port)
    uptime = _serial_queue.get_uptime() if _serial_queue else 0.0
    return {
        "status": "ok" if ollama_ok else "degraded",
        "queue_depth": queue_depth,
        "processing": processing,
        "ollama_reachable": ollama_ok,
        "uptime_seconds": int(uptime),
    }


@app.post("/gd/chat")
async def submit_chat(req: ChatRequest):
    """Submit inference request. Blocks until complete, returns result directly."""
    if _serial_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized.")

    depth = await _serial_queue.get_depth()
    max_depth = _serial_queue.max_depth
    if max_depth > 0 and depth >= max_depth:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({depth}/{max_depth}). Try again later or run 'gpu-access-router server restart'.",
        )

    inference_req = await _serial_queue.enqueue({
        "model": req.model,
        "messages": req.messages,
        "options": req.options,
        "timeout": req.timeout,
    })

    # Wait for inference to complete (event loop stays free for other requests)
    try:
        await asyncio.wait_for(inference_req.done_event.wait(), timeout=req.timeout + 10)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"Request timed out after {req.timeout}s")

    if inference_req.status == "error":
        raise HTTPException(status_code=500, detail=inference_req.error or "Inference error")
    if inference_req.status == "timeout":
        raise HTTPException(status_code=408, detail=inference_req.error or "Inference timed out")

    return inference_req.result


@app.get("/gd/queue/{request_id}")
async def get_queue_status(request_id: str):
    if _serial_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized.")
    status = await _serial_queue.get_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
    return status


@app.get("/gd/queue")
async def get_queue():
    if _serial_queue is None:
        return {"depth": 0, "max_depth": 0, "processing": False, "requests": []}
    depth = await _serial_queue.get_depth()
    processing = any(r.status == "processing" for r in _serial_queue._pending.values())
    requests = [
        {"request_id": r.request_id, "position": r.queue_position, "model": r.model}
        for r in _serial_queue._pending.values()
        if r.status == "waiting"
    ]
    return {"depth": depth, "max_depth": _serial_queue.max_depth, "processing": processing, "requests": requests}


@app.get("/gd/models")
async def list_models():
    url = f"http://localhost:{_ollama_port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")
    return {"models": data.get("models", [])}


# ---------------------------------------------------------------------------
# Ollama-compatible proxy endpoints (/api/*)
# ---------------------------------------------------------------------------

def _ensure_proxy_ready():
    """Check that httpx client and queue are available."""
    if httpx is None:
        raise HTTPException(
            status_code=503,
            detail="httpx not installed. Install with: pip install gpu-access-router[server]",
        )
    if _httpx_client is None:
        raise HTTPException(status_code=503, detail="Proxy client not initialized.")
    if _serial_queue is None:
        raise HTTPException(status_code=503, detail="Queue not initialized.")


async def _proxy_inference(request: Request, ollama_path: str):
    """
    Proxy an inference request to local Ollama with serial queue protection.

    Supports both streaming (NDJSON) and non-streaming (JSON) modes.
    All request fields are forwarded as-is (pass-through).
    """
    _ensure_proxy_ready()

    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = payload.get("model", "unknown")
    is_streaming = payload.get("stream", True)  # Ollama defaults to stream=true

    # Check queue depth before acquiring slot
    depth = await _serial_queue.get_depth()
    max_depth = _serial_queue.max_depth
    if max_depth > 0 and depth >= max_depth:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({depth}/{max_depth}). Try again later.",
        )

    # Acquire exclusive GPU slot (waits if another request is in progress)
    slot_id = await _serial_queue.acquire_slot(model=model)

    if is_streaming:
        return StreamingResponse(
            _stream_proxy(slot_id, ollama_path, body),
            media_type="application/x-ndjson",
        )
    else:
        # Non-streaming: forward, wait for full response, release slot
        try:
            resp = await _httpx_client.post(
                ollama_path,
                content=body,
                headers={"Content-Type": "application/json"},
            )
            return _make_json_response(resp)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail="Ollama request timed out")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Proxy error: {exc}")
        finally:
            _serial_queue.release_slot(slot_id)


async def _stream_proxy(slot_id: str, ollama_path: str, body: bytes):
    """Generator that streams NDJSON from Ollama, releasing the slot when done."""
    try:
        async with _httpx_client.stream(
            "POST",
            ollama_path,
            content=body,
            headers={"Content-Type": "application/json"},
        ) as upstream:
            if upstream.status_code != 200:
                error_body = await upstream.aread()
                yield json.dumps({"error": f"Ollama returned {upstream.status_code}: {error_body[:500].decode(errors='replace')}"}) + "\n"
                return

            async for line in upstream.aiter_lines():
                if line:
                    yield line + "\n"
    except httpx.ConnectError:
        yield json.dumps({"error": "Ollama service unavailable"}) + "\n"
    except httpx.TimeoutException:
        yield json.dumps({"error": "Ollama request timed out"}) + "\n"
    except Exception as exc:
        yield json.dumps({"error": f"Proxy error: {exc}"}) + "\n"
    finally:
        _serial_queue.release_slot(slot_id)


def _make_json_response(resp):
    """Convert httpx response to a FastAPI-compatible dict response."""
    try:
        return resp.json()
    except Exception:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text[:500] if resp.text else "Unknown Ollama error",
        )


@app.post("/api/generate")
async def proxy_generate(request: Request):
    """Ollama-compatible /api/generate proxy with serial queue protection."""
    return await _proxy_inference(request, "/api/generate")


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Ollama-compatible /api/chat proxy with serial queue protection."""
    return await _proxy_inference(request, "/api/chat")


@app.get("/api/tags")
async def proxy_tags():
    """Ollama-compatible /api/tags proxy (no queue needed — read-only)."""
    if httpx is None or _httpx_client is None:
        # Fallback to urllib if httpx not available
        url = f"http://localhost:{_ollama_port}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")

    try:
        resp = await _httpx_client.get("/api/tags")
        return resp.json()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unreachable: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ollama_reachable(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def run_server(host: str = "0.0.0.0", port: int = DEFAULT_API_PORT, reload: bool = False):
    uvicorn.run("gpu_access_router.server.api:app", host=host, port=port, reload=reload)
