"""
Generate inpainting using Stable Diffusion model without pretraining
"""
import os
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

#base directory
base_dir = 'data/digitized versions/Vies des saints'

#make directory for results
save_dir = f'{base_dir}/model_results/stablediffusion_oob'
os.makedirs(save_dir, exist_ok=True)

#read mutilated image and mask
init_image = PIL.Image.open(f'{base_dir}/mutilations/page_11_mutilated.jpeg').convert("RGB").resize((512, 512))
mask_image = PIL.Image.open(f'{base_dir}/masks/page_11_mask.jpeg').convert("RGB").resize((512, 512))

#define prompt
prompt = "text of medieval manuscript"

#generate inpainting
image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

#save inpainting
image.save(f"{save_dir}/page_11_inpainting.jpeg")