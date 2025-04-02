"""
AI Model Interface

A unified interface for interacting with various AI models including Google's Gemini 
and DeepSeek models via OpenRouter. This module provides a simple way to generate
text from different AI models using a consistent API.
"""

import openai
from google import genai
from typing import Dict, List, Optional, Union


class AIModelInterface:
    """
    A unified interface for generating text from various AI models.
    
    This class provides a consistent way to interact with different language models
    including Google's Gemini and DeepSeek models via OpenRouter.
    
    Attributes
    ----------
    model_configs : Dict[str, Dict]
        Configuration details for supported models.
    gemini_client : Optional[genai.Client]
        Client for Gemini (Google GenAI) if initialized.
    """

    def __init__(
        self, 
        gemini_api_key: Optional[str] = None, 
        openrouter_api_key: Optional[str] = None
    ):
        """
        Initialize the AI model interface with API keys.
        
        Parameters
        ----------
        gemini_api_key : Optional[str]
            API key for Google's Generative AI (Gemini) models.
        openrouter_api_key : Optional[str]
            API key for OpenRouter to access DeepSeek models.
        """
        self.model_configs = {
            # Example Gemini models
            "gemini-1.0-pro": {"type": "gemini", "model_name": "gemini-1.0-pro"},
            "gemini-1.5-pro": {"type": "gemini", "model_name": "gemini-1.5-pro"},
            "gemini-2.0-flash": {"type": "gemini", "model_name": "gemini-2.0-flash"},
            
            # Example DeepSeek models
            "deepseek-r1": {"type": "deepseek", "model_name": "deepseek/deepseek-r1:free"},
            "deepseek-chat-v3-0324": {
                "type": "deepseek", 
                "model_name": "deepseek/deepseek-chat-v3-0324:free"
            }
        }
        
        # Initialize Gemini client if key provided
        self.gemini_client = None
        if gemini_api_key:
            self.gemini_client = genai.Client(api_key=gemini_api_key)
        
        # Configure OpenRouter if key provided
        if openrouter_api_key:
            openai.api_base = "https://openrouter.ai/api/v1"
            openai.api_key = openrouter_api_key
    
    def generate_text(
        self, 
        model: str, 
        prompt: Union[str, List[Dict[str, str]]], 
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text using the specified AI model.
        
        Parameters
        ----------
        model : str
            Name of the model to use (e.g., "gemini-2.0-flash", "deepseek-chat-v3-0324").
        prompt : Union[str, List[Dict[str, str]]]
            Either a string prompt for simple queries or a list of message dictionaries
            for chat-based models in the format [{"role": "user", "content": "..."}].
        max_tokens : int, default=1000
            Maximum number of tokens to generate.
        
        Returns
        -------
        str
            Generated text response from the model.
        
        Raises
        ------
        ValueError
            If the model is not supported or required API key isn't set.
        RuntimeError
            If there's an error generating the response.
        """
        if model not in self.model_configs:
            raise ValueError(
                f"Model '{model}' not supported. "
                f"Available models: {list(self.model_configs.keys())}"
            )
        
        config = self.model_configs[model]
        model_type = config["type"]
        model_name = config["model_name"]
        
        try:
            # Handle Gemini models
            if model_type == "gemini":
                if not self.gemini_client:
                    raise ValueError(
                        "Gemini API key not provided. Initialize AIModelInterface with gemini_api_key."
                    )
                
                # Convert chat-style messages to a single string if needed
                if isinstance(prompt, list):
                    prompt = " ".join(
                        msg["content"] for msg in prompt if msg["role"] == "user"
                    )
                
                # Generate text with the Gemini client
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=[prompt]
                )
                return response.text
            
            # Handle DeepSeek models via OpenRouter
            elif model_type == "deepseek":
                if not openai.api_key:
                    raise ValueError(
                        "OpenRouter API key not provided. Initialize AIModelInterface with openrouter_api_key."
                    )
                
                # Format messages for chat completion
                if isinstance(prompt, str):
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = prompt
                
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                )
                return response['choices'][0]['message']['content']
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            raise RuntimeError(f"Error generating response with {model}: {str(e)}")

    def list_available_models(self) -> List[str]:
        """
        Get a list of all available models.
        
        Returns
        -------
        List[str]
            List of model names that can be used with this interface.
        """
        return list(self.model_configs.keys())


# Example usage
if __name__ == "__main__":
    # Initialize with (fake) API keys
    ai_interface = AIModelInterface(
        gemini_api_key="AIzaSyBXXXXXX",
        openrouter_api_key="sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXX"
    )
    
    # Example: Use a Gemini model
    gemini_response = ai_interface.generate_text(
        model="gemini-2.0-flash",
        prompt="How does AI work?"
    )
    print("Gemini response:")
    print(gemini_response)
    print("\n" + "-"*50 + "\n")
    
    # Example: Use a DeepSeek model
    deepseek_response = ai_interface.generate_text(
        model="deepseek-r1",
        prompt="What is the meaning of life?"
    )
    print("DeepSeek response:")
    print(deepseek_response)
