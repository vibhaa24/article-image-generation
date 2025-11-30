# article-image-generation
Image generation from articles using Stable Diffusion

This project takes three articles and generates realistic images based on their content using Stable Diffusion.
Each article has 2 images, so a total of 6 images are created.

How It Works

We read the articles.

We create simple and clear prompts that describe each article.

We use Stable Diffusion (public model) to generate realistic photos.

All images get saved in the /content folder in Google Colab.

Images Generated
Article 1

Article1_img1.png – Solar-powered African village at sunset

Article1_img2.png – Wind turbines near a coastal village

Article 2

Article2_img1.png – Youth street protest

Article2_img2.png – Candlelight vigil

Article 3

Article3_img1.png – Creative studio with digital screens

Article3_img2.png – Workstation close-up with monitors

Technologies Used

Python

Google Colab

Diffusers (Stable Diffusion)

PyTorch

Pillow

Steps to Run

Open Google Colab

Install required libraries

Copy and run the final code

Images will be saved in /content

Final Code Used
import os, torch
from diffusers import StableDiffusionPipeline

OUT = "/content"
os.makedirs(OUT, exist_ok=True)

MODEL = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device=="cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL,
    torch_dtype=dtype,
    safety_checker=None
).to(device)

pipe.enable_attention_slicing()

prompts = {
  "Article1_img1.png": "Realistic cinematic sunset photo of an African village with solar panels, families, children, sharp details, no text",
  "Article1_img2.png": "Wide-angle realistic coastal village with wind turbines, villagers repairing tools, clean lighting, no text",

  "Article2_img1.png": "Realistic youth protest with young people holding signs, natural expressions, no text",
  "Article2_img2.png": "Candlelight vigil with warm glow on faces, realistic details, no text",

  "Article3_img1.png": "Creative studio with holographic digital screens, designers working, cinematic look, no text",
  "Article3_img2.png": "Close-up workstation with monitors and hand pointing, bright clean details, no text"
}

NEG = "ugly, blurry, low quality, distorted face, text, watermark, extra limbs"

for i, (name, prompt) in enumerate(prompts.items(), start=1):
    seed = 1234 + i
    gen = torch.Generator(device=device).manual_seed(seed)
    img = pipe(prompt,
               height=720,
               width=512,
               num_inference_steps=35,
               guidance_scale=8,
               negative_prompt=NEG,
               generator=gen).images[0]
    img.save(os.path.join(OUT, name))

Created By
Vibha Pandey...

Created By

Vibha Pandey
