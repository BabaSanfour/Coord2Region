import openai

# Configure OpenAI SDK to use OpenRouter endpoint
openai.api_base = "https://openrouter.ai/api/v1"
openai.api_key = "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"

model = "deepseek-chat-v3-0324"

if model == "deepseek-r1":
    response = openai.ChatCompletion.create(
        model="deepseek/deepseek-r1:free",
        messages=[{"role": "user", "content": "What is the meaning of life?"}]
)
    print(response['choices'][0]['message']['content'])


elif model == "deepseek-chat-v3-0324":
    response = openai.ChatCompletion.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[{"role": "user", "content": "What is the meaning of life?"}]
    )
    print(response['choices'][0]['message']['content'])   
else:
    raise ValueError(f"Model {model} not supported")
