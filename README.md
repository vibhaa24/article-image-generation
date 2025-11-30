# Article Image Generation with Stable Diffusion

Generate realistic images from article content using Stable Diffusion.

## Overview

This project reads multiple articles and generates high-quality, realistic images that visually summarize each article. For each article, two unique images are created using Stable Diffusion (public model), totaling six images for three articles. Images are automatically saved in the `/content` folder on Google Colab.

## Workflow

1. **Article Analysis:** Read and understand each article.
2. **Prompt Creation:** Write simple, clear prompts describing the core ideas of each article.
3. **Image Generation:** Use Stable Diffusion to create realistic, cinematic images.
4. **Output:** Save all generated images in `/content` for easy access.

## Generated Images

**Article 1**
- `Article1_img1.png`: Solar-powered African village at sunset
- `Article1_img2.png`: Wind turbines near a coastal village

**Article 2**
- `Article2_img1.png`: Youth street protest
- `Article2_img2.png`: Candlelight vigil

**Article 3**
- `Article3_img1.png`: Creative studio with digital screens
- `Article3_img2.png`: Workstation close-up with monitors

## Technologies Used

- **Python**
- **Google Colab**
- **Diffusers (Stable Diffusion)**
- **PyTorch**
- **Pillow**

## How to Run

1. **Open** the project in Google Colab.
2. **Install** required libraries (`diffusers`, `torch`, `pillow`).
3. **Copy / Run** the final code block below.

All images will be saved automatically in `/content`.

---

## Example Code

```python
import os, torch
from diffusers import StableDiffusionPipeline

# Output directory
OUT = "/content"
os.makedirs(OUT, exist_ok=True)

# Model and device setup
MODEL = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL,
    torch_dtype=dtype,
    safety_checker=None,
).to(device)
pipe.enable_attention_slicing()

# Prompts for each image
prompts = {
    "Article1_img1.png": "Realistic cinematic sunset photo of an African village with solar panels, families, children, sharp details, no text",
    "Article1_img2.png": "Wide-angle realistic coastal village with wind turbines, villagers repairing tools, clean lighting, no text",
    "Article2_img1.png": "Realistic youth protest with young people holding signs, natural expressions, no text",
    "Article2_img2.png": "Candlelight vigil with warm glow on faces, realistic details, no text",
    "Article3_img1.png": "Creative studio with holographic digital screens, designers working, cinematic look, no text",
    "Article3_img2.png": "Close-up workstation with monitors and hand pointing, bright clean details, no text"
}
# Negative prompt example
NEG = "ugly, blurry, low quality, distorted face, text, watermark, extra limbs"

# Image generation loop
for i, (name, prompt) in enumerate(prompts.items(), start=1):
    seed = 1234 + i
    gen = torch.Generator(device=device).manual_seed(seed)
    img = pipe(
        prompt,
        height=720,
        width=512,
        num_inference_steps=35,
        guidance_scale=8,
        negative_prompt=NEG,
        generator=gen,
    ).images[0]
    img.save(os.path.join(OUT, name))
```

---

## Author 
Vibha Pandey

Created by **Vibha Pandey**

---
