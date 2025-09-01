"""Conditionally enable local HuggingFace text and image models
============================================================

"""

import sys

from coord2region.ai_model_interface import AIModelInterface


def main() -> None:
    use_text = "--text" in sys.argv
    use_image = "--image" in sys.argv

    cfg = {}
    if use_text:
        cfg["text_model"] = "distilgpt2"
    if use_image:
        cfg["image_model"] = "stabilityai/stable-diffusion-2"

    ai = AIModelInterface()
    ai.register_provider("huggingface_local", enabled=bool(cfg), **cfg)

    if use_text:
        print(ai.generate_text("distilgpt2", "Hello from local HF!"))
    if use_image:
        img = ai.generate_image(
            "stabilityai/stable-diffusion-2", "A brain rendered by Stable Diffusion"
        )
        with open("hf_local.png", "wb") as f:
            f.write(img)
        print("Saved image as hf_local.png")
    if not (use_text or use_image):
        print("Run with --text and/or --image to activate local models.")


if __name__ == "__main__":
    main()
