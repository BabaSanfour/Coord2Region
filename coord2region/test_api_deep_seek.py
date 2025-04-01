import openai

# Configure OpenAI SDK to use OpenRouter endpoint
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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"

response = openai.ChatCompletion.create(
    model="deepseek/deepseek-r1:free",
    messages=[{"role": "user", "content": "What is the meaning of life?"}]
)
print(response['choices'][0]['message']['content'])
