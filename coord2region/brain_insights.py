"""
Brain Insights Module

This module combines coordinate-to-study mapping with AI-powered analysis to provide
comprehensive insights about brain regions, including:
- Region identification and functional analysis
- Text summaries of brain regions based on neuroimaging literature
- Brain region visualization and image generation

Environment Variables
---------------------
GEMINI_API_KEY : Optional
    API key for Google's Generative AI (Gemini) models.
OPENROUTER_API_KEY : Optional
    API key for OpenRouter to access DeepSeek models.
OPENAI_API_KEY : Optional
    API key for OpenAI's GPT models.
ANTHROPIC_API_KEY : Optional
    API key for Anthropic's Claude models.
HUGGINGFACE_API_KEY or HUGGINGFACEHUB_API_TOKEN : Optional
    Access token for the HuggingFace Inference API.
DALLE_API_KEY : Optional
    API key for OpenAI's DALL·E image generation.
STABILITY_API_KEY : Optional
    API key for Stability AI image generation.
"""

import os
import json
import base64
import requests
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .coord2study import (
    get_studies_for_coordinate,
    prepare_datasets,
)
from .llm import (
    generate_region_image_prompt as _generate_region_image_prompt,
    generate_summary,
)
from .ai_model_interface import AIModelInterface

# Import from coord2region.py for atlas-based mapping
from .coord2region import (
    AtlasMapper, 
    MultiAtlasMapper
)
from .fetching import AtlasFetcher

logger = logging.getLogger(__name__)

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
        email_for_abstracts: Optional[str] = None,
        use_atlases: bool = True,
        atlas_names: Optional[List[str]] = None
    ):
        """
        Initialize the BrainInsights tool.
        
        Parameters
        ----------
        data_dir : str, default="nimare_data"
            Directory for NiMARE datasets and cached results
        gemini_api_key : Optional[str], default=None
            API key for Google's Generative AI (Gemini) models. If not provided,
            the ``GEMINI_API_KEY`` environment variable will be used.
        openrouter_api_key : Optional[str], default=None
            API key for OpenRouter to access DeepSeek models. If not provided,
            the ``OPENROUTER_API_KEY`` environment variable will be used.
        use_cached_dataset : bool, default=True
            Whether to use cached deduplicated dataset if available
        email_for_abstracts : Optional[str], default=None
            Email to use for PubMed abstract retrieval
        use_atlases : bool, default=True
            Whether to incorporate atlas-based anatomical labels
        atlas_names : Optional[List[str]], default=None
            List of atlas names to use. If None, uses ['harvard-oxford', 'juelich', 'aal']
        """
        self.data_dir = data_dir
        self.email = email_for_abstracts
        self.dataset = None
        self.use_atlases = use_atlases
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Cache directory for saving results
        self.cache_dir = os.path.join(data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")

        # Initialize AI model interface if keys are provided
        self.ai = None
        if gemini_api_key or openrouter_api_key:
            self.ai = AIModelInterface(
                gemini_api_key=gemini_api_key,
                openrouter_api_key=openrouter_api_key
            )
        
        # Load or prepare deduplicated dataset if requested
        if use_cached_dataset:
            self.dataset = prepare_datasets(data_dir)
        
        # Initialize atlas mappers if required
        self.atlases = {}
        self.multi_atlas = None
        if use_atlases:
            self.atlas_names = atlas_names or ['harvard-oxford', 'juelich', 'aal']
            self._init_atlases(self.atlas_names)
    
    def _init_atlases(self, atlas_names: List[str]):
        """
        Initialize atlas mappers.
        
        Parameters
        ----------
        atlas_names : List[str]
            List of atlas names to initialize
        """
        try:
            # Create atlas fetcher
            fetcher = AtlasFetcher()
            
            # Initialize each atlas
            for atlas_name in atlas_names:
                try:
                    atlas_data = fetcher.fetch_atlas(atlas_name)
                    self.atlases[atlas_name] = AtlasMapper(
                        name=atlas_name,
                        vol=atlas_data['vol'],
                        hdr=atlas_data['hdr'],
                        labels=atlas_data['labels']
                    )
                except Exception as e:
                    logger.warning("Failed to load atlas %s: %s", atlas_name, e)
            
            # Create multi-atlas mapper if atlases were loaded
            if self.atlases:
                self.multi_atlas = MultiAtlasMapper([mapper for mapper in self.atlases.values()])
            
        except Exception as e:
            logger.warning("Failed to initialize atlases: %s", e)
            self.use_atlases = False
    
    def get_atlas_labels(self, coordinate: Union[List[float], Tuple[float, float, float]]) -> Dict[str, str]:
        """
        Get anatomical labels for a coordinate from all available atlases.
        
        Parameters
        ----------
        coordinate : Union[List[float], Tuple[float, float, float]]
            MNI coordinate [x, y, z]
            
        Returns
        -------
        Dict[str, str]
            Dictionary mapping atlas names to region labels
        """
        if not self.use_atlases or not self.atlases:
            return {}
        
        atlas_labels = {}
        
        # Try using multi-atlas mapper first
        if self.multi_atlas:
            try:
                multi_results = self.multi_atlas.mni_to_region_names(coordinate)
                return multi_results
            except Exception as e:
                logger.warning("Multi-atlas mapping failed: %s", e)
        
        # Fall back to individual atlases if multi-atlas fails
        for atlas_name, mapper in self.atlases.items():
            try:
                region_name = mapper.mni_to_region_name(coordinate)
                atlas_labels[atlas_name] = region_name
            except Exception as e:
                logger.warning("Failed to map coordinate with atlas %s: %s", atlas_name, e)
        
        return atlas_labels
    
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
        if self.dataset is None:
            self.dataset = prepare_datasets(self.data_dir)
        if not self.dataset:
            return []

        studies = get_studies_for_coordinate(
            {"Combined": self.dataset},
            coordinate,
            radius=radius,
            email=self.email,
        )
        return studies
    
    def get_region_summary(
        self,
        coordinate: Union[List[float], Tuple[float, float, float]],
        summary_type: str = "summary",
        model: str = "gemini-2.0-flash",
        include_atlas_labels: bool = True,
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of a brain region at the specified coordinate.
        
        Uses neuroimaging literature, atlas labels, and AI to provide detailed information about the
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
        include_atlas_labels : bool, default=True
            Whether to include atlas labels in the prompt
        prompt_template : str, optional
            Custom template to override the default prompt generation. See
            :func:`generate_llm_prompt` for placeholder details.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "coordinate": The input coordinate
            - "studies": List of studies found
            - "summary": Generated text summary
            - "studies_count": Number of studies found
            - "atlas_labels": Dictionary of atlas labels (if include_atlas_labels is True)
        """
        # Check if we can use the AI model
        if not self.ai:
            raise ValueError("AI model interface not initialized. Provide API keys when initializing BrainInsights.")
        
        # Try to load from cache first
        cache_suffix = "_with_atlas" if include_atlas_labels and self.use_atlases else ""
        cache_key = f"summary_{coordinate[0]}_{coordinate[1]}_{coordinate[2]}_{summary_type}_{model}{cache_suffix}"
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

        # Get atlas labels if requested
        atlas_labels: Dict[str, str] = {}
        if include_atlas_labels and self.use_atlases:
            atlas_labels = self.get_atlas_labels(coordinate)

        if not studies and not atlas_labels:
            return {
                "coordinate": coordinate,
                "studies": [],
                "summary": f"No studies or atlas labels found for MNI coordinate {coordinate}.",
                "studies_count": 0,
                "atlas_labels": {}
            }

        # Generate summary using centralized LLM service
        summary = generate_summary(
            self.ai,
            studies,
            coordinate,
            summary_type=summary_type,
            model=model,
            atlas_labels=atlas_labels if include_atlas_labels else None,
            prompt_template=prompt_template,
        )
        
        # Prepare result
        result = {
            "coordinate": coordinate,
            "studies": studies,
            "summary": summary,
            "studies_count": len(studies),
            "atlas_labels": atlas_labels
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
                "studies_count": len(studies),
                "atlas_labels": atlas_labels
            }
            json.dump(serializable_result, f, indent=2)
        
        return result
    
    def generate_region_image_prompt(
        self,
        coordinate: Union[List[float], Tuple[float, float, float]],
        image_type: str = "anatomical",
        include_atlas_labels: bool = True
    ) -> str:
        """Generate a prompt for creating images of brain regions."""

        region_info = self.get_region_summary(
            coordinate,
            summary_type="region_name",
            include_atlas_labels=include_atlas_labels,
        )

        return _generate_region_image_prompt(
            coordinate,
            region_info,
            image_type=image_type,
            include_atlas_labels=include_atlas_labels,
        )
    
    def generate_region_image(
        self, 
        coordinate: Union[List[float], Tuple[float, float, float]],
        image_type: str = "anatomical",
        dalle_api_key: Optional[str] = None,
        stability_api_key: Optional[str] = None,
        save_image: bool = True,
        include_atlas_labels: bool = True
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
            OpenAI API key for DALL-E image generation. If not provided,
            the ``DALLE_API_KEY`` environment variable will be used.
        stability_api_key : Optional[str], default=None
            Stability AI API key for image generation. If not provided,
            the ``STABILITY_API_KEY`` environment variable will be used.
        save_image : bool, default=True
            Whether to save the generated image to disk
        include_atlas_labels : bool, default=True
            Whether to include atlas labels in the prompt generation
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - "prompt": The prompt used for generation
            - "image_url": URL of the generated image (if using DALL-E)
            - "image_path": Path to saved image (if save_image is True)
        """
        dalle_api_key = dalle_api_key or os.environ.get("DALLE_API_KEY")
        stability_api_key = stability_api_key or os.environ.get("STABILITY_API_KEY")

        # Generate the prompt for image creation
        prompt = self.generate_region_image_prompt(
            coordinate,
            image_type,
            include_atlas_labels=include_atlas_labels
        )

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
    model: str = "gemini-2.0-flash",
    include_atlas_labels: bool = True
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
    include_atlas_labels : bool, default=True
        Whether to include atlas labels in the analysis
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary information about the brain region
    """
    brain = BrainInsights(
        gemini_api_key=gemini_api_key,
        openrouter_api_key=openrouter_api_key,
        use_atlases=include_atlas_labels
    )
    return brain.get_region_summary(
        coordinate, 
        summary_type, 
        model, 
        include_atlas_labels=include_atlas_labels
    )


# Example usage
if __name__ == "__main__":
    import logging

    # Configure logging to output to the console
    logging.basicConfig(level=logging.INFO)

    # Initialize with API keys from environment variables
    insights = BrainInsights(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        email_for_abstracts="snesmaeil@gmail.com",
        use_atlases=True,
        atlas_names=['harvard-oxford', 'juelich', 'aal']
    )

    # Example coordinate
    coordinate = [30, 22, -8]

    # Get atlas labels for the coordinate
    atlas_labels = insights.get_atlas_labels(coordinate)
    logger.info("Atlas Labels for coordinate %s:", coordinate)
    for atlas, label in atlas_labels.items():
        logger.info("  %s: %s", atlas, label)
    logger.info("")

    # Get region summary with atlas labels
    result = insights.get_region_summary(
        coordinate=coordinate,
        summary_type="summary",
        model="gemini-2.0-flash",
        include_atlas_labels=True
    )

    logger.info("Brain Region Summary for coordinate %s:", coordinate)
    logger.info("Found %s relevant studies", result['studies_count'])
    logger.info("\n" + "="*80)
    logger.info("%s", result["summary"])
    logger.info("="*80)

    # Generate image prompt with atlas labels
    image_prompt = insights.generate_region_image_prompt(
        coordinate=coordinate,
        image_type="anatomical",
        include_atlas_labels=True
    )

    logger.info("\nImage Generation Prompt (with atlas labels):")
    logger.info("-"*80)
    logger.info("%s", image_prompt)
    logger.info("-"*80)

    # Note: Uncomment to actually generate images if you have API keys
    # dalle_api_key = os.environ.get("DALLE_API_KEY")
    # image_result = insights.generate_region_image(
    #     coordinate=coordinate,
    #     image_type="anatomical",
    #     dalle_api_key=dalle_api_key,
    #     include_atlas_labels=True
    # )
    # if "image_path" in image_result:
    #     logger.info("Image saved to: %s", image_result['image_path'])
