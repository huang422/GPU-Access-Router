from gpu_access_router import GPURouter

router = GPURouter()

response = router.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "用繁體中文解說機器學習"}]
)
print(response.message.content)