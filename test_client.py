"""Test client — demonstrates unified usage across all environments.

Routing is controlled entirely by environment variables; no code changes needed:

  # Local Ollama only
  GPU_ROUTER_ROUTING_MODE=local python test_client.py

  # Remote GPU (auto, with fallback to local small model on failure)
  GPU_ROUTER_SERVER_IP=<ip> GPU_ROUTER_ROUTING_MODE=auto \
  GPU_ROUTER_FALLBACK_MODEL=qwen3.5:9b python test_client.py

  # Force remote, no fallback
  GPU_ROUTER_SERVER_IP=<ip> GPU_ROUTER_ROUTING_MODE=remote python test_client.py
"""

from gpu_access_router import ollama

response = ollama.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "用繁體中文一句話解釋機器學習"}]
)
print(response.message.content)
