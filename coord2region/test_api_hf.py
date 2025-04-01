import requests

# MODEL_ID="google/gemma-3-1b-it"
MODEL_ID="meta-llama/Llama-2-7b"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"}
    # "x-use-cache": "false"}
data = {
    "inputs": "what is the capital of France? "
}
response = requests.post(API_URL, headers=headers, json=data)
print(response.json())