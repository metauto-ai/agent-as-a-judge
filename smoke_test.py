import os
from litellm import completion

model = os.getenv("DEFAULT_LLM")
print("Model:", model)
resp = completion(
    model=model,
    messages=[{"role": "user", "content": "Say hi in one short line."}],
    # 如果你没用 .env 加载，亦可显式传 api_base/api_key：
    api_base="https://api.lingleap.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
)
print(resp["choices"][0]["message"]["content"])
