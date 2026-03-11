from gpu_directer import GPURouter

router = GPURouter()

response = router.chat(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Explain what a GPU is in one sentence."}]
)
print(response.message.content)