"""FastAPI server exposing /gd/* endpoints with serial queue."""

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    from pydantic import BaseModel
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-directer[server]"
    ) from exc

try:
    import ollama as _ollama
except ImportError as exc:
    raise ImportError(
        "Server dependencies not installed. Install with: pip install gpu-directer[server]"
    ) from exc

from gpu_directer.core.constants import DEFAULT_API_PORT, DEFAULT_PORT, DEFAULT_QUEUE_DEPTH, DEFAULT_TIMEOUT
from gpu_directer.server.queue import SerialQueue

app = FastAPI(title="GPU Directer Server", version="0.1.0")

_serial_queue: Optional[SerialQueue] = None
_ollama_port: int = DEFAULT_PORT


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    options: Dict[str, Any] = {}
    timeout: int = DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _serial_queue, _ollama_port
    from gpu_directer import config as cfg_mod
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
    processing = any(
        r.status == "processing"
        for r in (_serial_queue._pending.values() if _serial_queue else [])
    )
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
            detail=f"Queue full ({depth}/{max_depth}). Try again later or run 'gpu-directer server restart'.",
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
# Helpers
# ---------------------------------------------------------------------------

def _check_ollama_reachable(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/tags", timeout=3):
            return True
    except Exception:
        return False


def run_server(host: str = "0.0.0.0", port: int = DEFAULT_API_PORT, reload: bool = False):
    uvicorn.run("gpu_directer.server.api:app", host=host, port=port, reload=reload)
