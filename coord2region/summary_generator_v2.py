import requests

def generate_summary(coordinate, studies, mode="api", api_token=None, max_new_tokens=150):
    """
    Generate a summary using LLaMA 2 based on the given coordinate and studies.
    
    Parameters:
      - coordinate: List or tuple of three floats representing the MNI coordinate.
      - studies: List of dictionaries with study details (e.g., keys "title" and "abstract").
      - mode: Either "local" or "api". If "local", the model is loaded on your machine.
              If "api", the Hugging Face Inference API is used.
      - api_token: Required if mode=="api". Your Hugging Face access token with inference permissions.
      - max_new_tokens: Maximum number of tokens to generate.
    
    Returns:
      A string with the generated summary.
    """
    # Build a prompt that lists the coordinate and the study details.
    prompt = f"Based on the following studies for MNI coordinate {coordinate}:\n\n"
    for study in studies:
        title = study.get("title", "No title provided")
        abstract = study.get("abstract", "No abstract provided")
        prompt += f"- Title: {title}\n  Abstract: {abstract}\n\n"
    prompt += "Provide a concise summary highlighting the main findings and common insights."
    
    if mode.lower() == "local":
        # Local mode: Load the model on your computer.
        # Note: You must have PyTorch (or TensorFlow/Flax) installed and sufficient resources.
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        # Specify the model ID; adjust if you want a different LLaMA 2 variant.
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        
        # Generate and return the summary.
        generated = summarizer(prompt)[0]['generated_text']
        return generated

    elif mode.lower() == "api":
        # from huggingface_hub import InferenceClient
        # # API mode: Use Hugging Face Inference API.
        # api_token="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"
        # model_id = "meta-llama/Llama-2-7b-chat-hf"
        # client = InferenceClient(provider="hf-inference", api_key="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")

        # completion = client.chat.completions.create(model=model_id,messages=[{"role": "user", "content": prompt}], max_tokens=500)        
        # # Parse and return the generated text.
        # generated_text = print(completion.choices[0].message)
        # return generated_text
        # Use the text-generation endpoint (free tier)
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        api_token="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"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code}, {response.text}")
        # The response is a list of generation results.
        generated_text = response.json()[0].get("generated_text", "")
        return generated_text    
    else:
        raise ValueError("Invalid mode. Please choose either 'local' or 'api'.")

if __name__ == '__main__':
    # Example studies for testing.
    test_studies = [
        {"id": "123", "title": "Study on Neural Mechanisms",
         "abstract": "This study investigates neural mechanisms underlying motor control."},
        {"id": "456", "title": "Study on Brain Mapping",
         "abstract": "This research maps out brain regions associated with movement coordination."}
    ]
    test_coordinate = [30, 22, -8]
    
    # Choose your mode here: set mode to "local" or "api".
    mode = "api"  # Change to "local" to run the model locally.
    
    if mode.lower() == "api":
        # Replace with your actual Hugging Face API token.
        hf_api_token = "YOUR_HF_API_TOKEN"
        summary = generate_summary(test_coordinate, test_studies, mode=mode, api_token=hf_api_token)
    else:
        summary = generate_summary(test_coordinate, test_studies, mode=mode)
    
    print("Generated Summary:\n", summary)
