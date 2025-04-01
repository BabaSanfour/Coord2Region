import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def generate_summary(coordinate, studies, max_new_tokens=200):
    """
    Generate a summary using LLaMA 2 based on the given coordinate and studies.

    :param coordinate: List or tuple of three floats representing the MNI coordinate.
    :param studies: List of study dictionaries (each with keys like "title" and "abstract").
    :param max_new_tokens: Maximum number of tokens to generate.
    :return: Generated summary string.
    """
    # Create a prompt that includes the coordinate and study details.
    prompt = f"Based on the following studies found for MNI coordinate {coordinate}:\n\n"
    for study in studies:
        title = study.get("title", "No title provided")
        abstract = study.get("abstract", "No abstract provided")
        prompt += f"- Title: {title}\n  Abstract: {abstract}\n\n"
    prompt += "Please provide a concise summary highlighting the common findings and insights related to this coordinate."

    # Specify the model. (You may change to 'meta-llama/Llama-2-70b-chat-hf' if you have sufficient GPU memory.)
    model_id = "meta-llama/Llama-2-7b-hf"

    # Load the tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    
    # Create a text-generation pipeline.
    summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
    
    # Generate the summary.
    generated = summarizer(prompt)[0]['generated_text']
    
    # Depending on the prompt design, you might want to extract only the summary part.
    # Here we assume the full generated text is the summary.
    return generated

if __name__ == '__main__':
    # Test example: define dummy studies and a coordinate.
    test_studies = [
        {"id": "123", "title": "Study on neural mechanisms", "abstract": "This study investigates neural mechanisms underlying motor control."},
        {"id": "456", "title": "Study on brain mapping", "abstract": "This research maps out brain regions associated with movement coordination."}
    ]
    test_coordinate = [30, 22, -8]
    summary = generate_summary(test_coordinate, test_studies)
    print("Generated Summary:\n", summary)
