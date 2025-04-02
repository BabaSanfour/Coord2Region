"""
Brain Insights Module

This module combines coordinate-to-study mapping with AI-powered analysis to provide
comprehensive insights about brain regions, including:
- Region identification and functional analysis
- Text summaries of brain regions based on neuroimaging literature
- Brain region visualization and image generation
"""

import os
import json
import base64
import requests
from typing import Any, Dict, List, Optional, Tuple, Union

from .coord2study import (
    get_studies_for_coordinate,
    get_studies_for_coordinate_dedup,
    generate_llm_prompt,
    load_deduplicated_dataset
)
from .ai_model_interface import AIModelInterface


class BrainInsights:
    """
    A comprehensive tool for analyzing brain coordinates using neuroimaging literature and AI.
    
    This class combines coordinate-to-study mapping with AI text generation to provide
    detailed insights about brain regions, including anatomical labels, functional roles,
    and visualizations.
    """
    
    def __init__(
        self,
        data_dir: str = "nimare_data",
        gemini_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        use_cached_dataset: bool = True,
        email_for_abstracts: Optional[str] = None
    ):
        """
        Initialize the BrainInsights tool.
        
        Parameters
        ----------
        data_dir : str, default="nimare_data"
            Directory for NiMARE datasets and cached results
        gemini_api_key : Optional[str], default=None
            API key for Google's Generative AI (Gemini) models
        openrouter_api_key : Optional[str], default=None
            API key for OpenRouter to access DeepSeek models
        use_cached_dataset : bool, default=True
            Whether to use cached deduplicated dataset if available
        email_for_abstracts : Optional[str], default=None
            Email to use for PubMed abstract retrieval
        """
        self.data_dir = data_dir
        self.email = email_for_abstracts
        self.dataset = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Cache directory for saving results
        self.cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize AI model interface if keys are provided
        self.ai = None
        if gemini_api_key or openrouter_api_key:
            self.ai = AIModelInterface(
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key
            )
        
        # Load deduplicated dataset if available
        if use_cached_dataset:
            dedup_path = os.path.join(data_dir, "deduplicated_dataset.pkl.gz")
            if os.path.exists(dedup_path):
                self.dataset = load_deduplicated_dataset(dedup_path)
    
    def get_region_studies(
        self, 
        coordinate: Union[List[float], Tuple[float, float, float]],
        radius: float = 0
    ) -> List[Dict[str, Any]]:
        """
        Get studies reporting activation at the specified MNI coordinate.
        
        Parameters
        ----------
        coordinate : Union[List[float], Tuple[float, float, float]]
            MNI coordinate [x, y, z]
        radius : float, default=0
            Search radius in mm around the coordinate (0 for exact match)
            
        Returns
        -------
        List[Dict[str, Any]]
            List of study metadata dictionaries
        """
        if self.dataset:
            # Use deduplicated dataset if available
            studies = get_studies_for_coordinate_dedup(
                self.dataset,
                coordinate,
                email=self.email
            )
        else:
            # Otherwise, fetch from online sources (would need to implement fetch_datasets here)
            from .coord2study import fetch_datasets
            datasets = fetch_datasets(self.data_dir)
            studies = get_studies_for_coordinate(datasets, coordinate, email=self.email)
        
        return studies
    
    def get_region_summary(
        self,
        coordinate: Union[List[float], Tuple[float, float, float]],
        summary_type: str = "summary",
        model: str = "gemini-2.0-flash"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of a brain region at the specified coordinate.
        
        Uses neuroimaging literature and AI to provide detailed information about the
        brain region, including anatomical labels and functional roles.
        
        Parameters
        ----------
        coordinate : Union[List[float], Tuple[float, float, float]]
            MNI coordinate [x, y, z]
        summary_type : str, default="summary"
            Type of summary to generate:
            - "summary": General overview
            - "region_name": Focus on anatomical labels
            - "function": Detailed functional analysis
        model : str, default="gemini-2.0-flash"
            AI model to use for generating the summary
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "coordinate": The input coordinate
            - "studies": List of studies found
            - "summary": Generated text summary
            - "studies_count": Number of studies found
        """
        # Check if we can use the AI model
        if not self.ai:
            raise ValueError("AI model interface not initialized. Provide API keys when initializing BrainInsights.")
        
        # Try to load from cache first
        cache_key = f"summary_{coordinate[0]}_{coordinate[1]}_{coordinate[2]}_{summary_type}_{model}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                # If loading fails, continue with generating a new summary
                pass
        
        # Get studies for the coordinate
        studies = self.get_region_studies(coordinate)
        
        if not studies:
            return {
                "coordinate": coordinate,
                "studies": [],
                "summary": f"No studies found for MNI coordinate {coordinate}.",
                "studies_count": 0
            }
        
        # Generate prompt for the AI model
        prompt = generate_llm_prompt(studies, coordinate, prompt_type=summary_type)
        
        # Generate summary using AI model
        summary = self.ai.generate_text(model=model, prompt=prompt)
        
        # Prepare result
        result = {
            "coordinate": coordinate,
            "studies": studies,
            "summary": summary,
            "studies_count": len(studies)
        }
        
        # Cache the result
        with open(cache_file, 'w') as f:
            # Convert to JSON-serializable format (remove complex objects)
            serializable_result = {
                "coordinate": coordinate,
                "studies": [
                    {
                        "id": study.get("id", ""),
                        "title": study.get("title", ""),
                        # Limit abstract length for cache file size
                        "abstract": study.get("abstract", "")[:500] + "..." 
                            if study.get("abstract") and len(study.get("abstract", "")) > 500 
                            else study.get("abstract", "")
                    }
                    for study in studies
                ],
                "summary": summary,
                "studies_count": len(studies)
            }
            json.dump(serializable_result, f, indent=2)
        
        return result
    
    def generate_region_image_prompt(
        self,
        coordinate: Union[List[float], Tuple[float, float, float]],
        image_type: str = "anatomical"
    ) -> str:
        """
        Generate a prompt for creating images of brain regions.
        
        Parameters
        ----------
        coordinate : Union[List[float], Tuple[float, float, float]]
            MNI coordinate [x, y, z]
        image_type : str, default="anatomical"
            Type of image to generate:
            - "anatomical": Anatomical visualization
            - "functional": Functional activity visualization
            - "schematic": Schematic diagram
            - "artistic": Artistic representation
            
        Returns
        -------
        str
            A detailed prompt for image generation models
        """
        # First get a summary to identify the region
        region_info = self.get_region_summary(coordinate, summary_type="region_name")
        summary = region_info["summary"]
        
        # Extract likely region names from the summary (this is a simple approach)
        # A more sophisticated approach would use NLP to extract the region names
        first_paragraph = summary.split("\n\n")[0] if "\n\n" in summary else summary
        
        # Base prompt template
        if image_type == "anatomical":
            prompt = f"""Create a detailed anatomical illustration of the brain region at MNI coordinate {coordinate}.
Based on neuroimaging studies, this location corresponds to: {first_paragraph}
Show a clear, labeled anatomical visualization with the specific coordinate marked.
Include surrounding brain structures for context. Use a professional medical illustration style with
accurate colors and textures of brain tissue."""
            
        elif image_type == "functional":
            prompt = f"""Create a functional brain activation visualization showing activity at MNI coordinate {coordinate}.
This region corresponds to: {first_paragraph}
Show the activation as a heat map or colored overlay on a standardized brain template.
Use a scientific visualization style similar to fMRI results in neuroscience publications,
with the activation at the specified coordinate clearly highlighted."""
            
        elif image_type == "schematic":
            prompt = f"""Create a schematic diagram of brain networks involving the region at MNI coordinate {coordinate}.
This coordinate corresponds to: {first_paragraph}
Show this region as a node in its relevant brain networks, with connections to other regions.
Use a simplified, clean diagram style with labeled regions and connection lines indicating functional
or structural connectivity. Include a small reference brain to indicate the location."""
            
        elif image_type == "artistic":
            prompt = f"""Create an artistic visualization of the brain region at MNI coordinate {coordinate}.
This region is: {first_paragraph}
Create an artistic interpretation that conveys the function of this region through metaphorical
or abstract elements, while still maintaining scientific accuracy in the brain anatomy.
Balance creativity with neuroscientific precision."""
            
        else:
            prompt = f"""Create a clear visualization of the brain region at MNI coordinate {coordinate}.
Based on neuroimaging studies, this region corresponds to: {first_paragraph}
Show this region clearly marked on a standard brain template with proper anatomical context."""
        
        return prompt
    
    def generate_region_image(
        self, 
        coordinate: Union[List[float], Tuple[float, float, float]],
        image_type: str = "anatomical",
        dalle_api_key: Optional[str] = None,
        stability_api_key: Optional[str] = None,
        save_image: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an image of a brain region using an image generation API.
        
        This requires either a DALL-E API key or Stability API key.
        
        Parameters
        ----------
        coordinate : Union[List[float], Tuple[float, float, float]]
            MNI coordinate [x, y, z]
        image_type : str, default="anatomical"
            Type of image to generate (anatomical, functional, schematic, artistic)
        dalle_api_key : Optional[str], default=None
            OpenAI API key for DALL-E image generation
        stability_api_key : Optional[str], default=None
            Stability AI API key for image generation
        save_image : bool, default=True
            Whether to save the generated image to disk
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "prompt": The prompt used for generation
            - "image_url": URL of the generated image (if using DALL-E)
            - "image_path": Path to saved image (if save_image is True)
        """
        # Generate the prompt for image creation
        prompt = self.generate_region_image_prompt(coordinate, image_type)
        
        result = {
            "prompt": prompt,
            "coordinate": coordinate,
            "image_type": image_type,
        }
        
        # Check if we have any API keys
        if not dalle_api_key and not stability_api_key:
            result["error"] = "No image generation API keys provided."
            return result
        
        # Try to generate with DALL-E first if that key is provided
        if dalle_api_key:
            try:
                import openai
                openai.api_key = dalle_api_key
                
                response = openai.Image.create(
                    prompt=prompt,
                    n=1,
                    size="1024x1024"
                )
                
                image_url = response['data'][0]['url']
                result["image_url"] = image_url
                
                # Save the image if requested
                if save_image:
                    img_dir = os.path.join(self.cache_dir, "images")
                    os.makedirs(img_dir, exist_ok=True)
                    
                    img_filename = f"region_{coordinate[0]}_{coordinate[1]}_{coordinate[2]}_{image_type}.png"
                    img_path = os.path.join(img_dir, img_filename)
                    
                    # Download the image
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        with open(img_path, 'wb') as f:
                            f.write(img_response.content)
                        result["image_path"] = img_path
                
                return result
                
            except Exception as e:
                # If DALL-E fails, try Stability API if available
                result["dalle_error"] = str(e)
        
        # Try Stability API if DALL-E failed or wasn't used
        if stability_api_key:
            try:
                url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
                
                headers = {
                    "Authorization": f"Bearer {stability_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                
                payload = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "samples": 1,
                    "steps": 30,
                }
                
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Get base64 encoded image
                    image_b64 = data["artifacts"][0]["base64"]
                    
                    # Save the image if requested
                    if save_image:
                        img_dir = os.path.join(self.cache_dir, "images")
                        os.makedirs(img_dir, exist_ok=True)
                        
                        img_filename = f"region_{coordinate[0]}_{coordinate[1]}_{coordinate[2]}_{image_type}.png"
                        img_path = os.path.join(img_dir, img_filename)
                        
                        # Save the base64 decoded image
                        with open(img_path, 'wb') as f:
                            f.write(base64.b64decode(image_b64))
                        result["image_path"] = img_path
                    
                    return result
                else:
                    result["stability_error"] = f"API error: {response.status_code} - {response.text}"
                    
            except Exception as e:
                result["stability_error"] = str(e)
        
        # If we got here, both APIs failed or weren't available
        if "dalle_error" in result and "stability_error" in result:
            result["error"] = "Both image generation APIs failed."
        
        return result


# Function to quickly get brain region insights from a coordinate
def get_brain_insights(
    coordinate: Union[List[float], Tuple[float, float, float]],
    gemini_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    summary_type: str = "summary",
    model: str = "gemini-2.0-flash"
) -> Dict[str, Any]:
    """
    Quickly get AI-generated insights about a brain region at a specific coordinate.
    
    This is a convenience function for one-off analyses without creating a BrainInsights instance.
    
    Parameters
    ----------
    coordinate : Union[List[float], Tuple[float, float, float]]
        MNI coordinate [x, y, z]
    gemini_api_key : Optional[str], default=None
        API key for Google's Generative AI (Gemini) models
    openrouter_api_key : Optional[str], default=None
        API key for OpenRouter to access DeepSeek models
    summary_type : str, default="summary"
        Type of summary to generate (summary, region_name, function)
    model : str, default="gemini-2.0-flash"
        AI model to use for generating the summary
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary information about the brain region
    """
    brain = BrainInsights(
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key
    )
    return brain.get_region_summary(coordinate, summary_type, model)


# Example usage
if __name__ == "__main__":
    # Initialize with API keys
    insights = BrainInsights(
        gemini_api_key="AIzaSyBXQFQ4PbB29BteSFs1zDq5dD8o8YkbKxg",
        openrouter_api_key="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",
        email_for_abstracts="snesmaeil@gmail.com"
    )
    
    # Example coordinate
    coordinate = [30, 22, -8]
    
    # Get region summary
    result = insights.get_region_summary(
        coordinate=coordinate,
        summary_type="summary", 
        model="gemini-2.0-flash"
    )
    
    print(f"Brain Region Summary for coordinate {coordinate}:")
    print(f"Found {result['studies_count']} relevant studies")
    print("\n" + "="*80)
    print(result["summary"])
    print("="*80)
    
    # Generate image prompt
    image_prompt = insights.generate_region_image_prompt(
        coordinate=coordinate,
        image_type="anatomical"
    )
    
    print("\nImage Generation Prompt:")
    print("-"*80)
    print(image_prompt)
    print("-"*80)
    
    # Note: Uncomment to actually generate images if you have API keys
    # dalle_api_key = "your_dalle_api_key"
    # image_result = insights.generate_region_image(
    #     coordinate=coordinate,
    #     image_type="anatomical",
    #     dalle_api_key=dalle_api_key
    # )
    # if "image_path" in image_result:
    #     print(f"Image saved to: {image_result['image_path']}") 