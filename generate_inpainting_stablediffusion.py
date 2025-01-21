"""
Generate inpainting using Stable Diffusion model without pretraining
"""
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm
import PIL.Image

# Load the inpainting pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

# Base directory
base_dir = 'data/digitized versions/Vies des saints'

# Make directory for results
save_dir = f'{base_dir}/model_results/stablediffusion_oob'
os.makedirs(save_dir, exist_ok=True)

# Define prompt
prompt = "text of medieval manuscript"

# Loop through all images in the mutilations directory
mutilations_dir = f'{base_dir}/mutilations'
masks_dir = f'{base_dir}/masks'
excision_dir = f'{base_dir}/excisions'

# Get list of files
files = [f for f in os.listdir(mutilations_dir) if f.endswith(".jpeg")]

# Loop through all images in the mutilations directory with progress bar
for i, filename in enumerate(tqdm(files, desc="Processing images", unit="image")):
    if i >= 10:
        break
    mutilated_image_path = os.path.join(mutilations_dir, filename)
    mask_image_path = os.path.join(masks_dir, filename.replace("mutilated", "mask"))
    excision_image_path = os.path.join(excision_dir, filename.replace("mutilated", "excised"))

    try:
        init_image = PIL.Image.open(mutilated_image_path).convert("RGB")
        mask_image = PIL.Image.open(mask_image_path).convert("RGB")
        excision_image = PIL.Image.open(excision_image_path).convert("RGB")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        continue

    # Generate the inpainting
    try:
        result = pipeline(prompt=prompt, image=init_image, mask_image=mask_image)
        inpainted_image = result.images[0]
    except Exception as e:
        print(f"Error during inpainting for {filename}: {e}")
        continue

    # Save the inpainting result
    output_path = os.path.join(save_dir, filename.replace("mutilated", "inpainting"))
    inpainted_image.save(output_path)
    print(f"Inpainted image saved to {output_path}")

    # # Create a comparison image
    # comparison_image = PIL.Image.new('RGB', (init_image.width * 3, init_image.height))
    # comparison_image.paste(init_image, (0, 0))
    # comparison_image.paste(excision_image, (init_image.width, 0))
    # comparison_image.paste(inpainted_image, (init_image.width * 2, 0))

    # # Save the comparison image
    # comparison_output_path = os.path.join(save_dir, filename.replace("mutilated", "comparison"))
    # comparison_image.save(comparison_output_path)
    # print(f"Comparison image saved to {comparison_output_path}")