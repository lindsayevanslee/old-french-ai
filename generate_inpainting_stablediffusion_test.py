"""
Generate inpainting using Stable Diffusion model without pretraining. Run on one image at a time.
"""
import os
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import math

def get_dimensions_divisible_by_8(width, height):
    """Calculate new dimensions that are divisible by 8 while maintaining aspect ratio."""
    new_width = math.ceil(width / 8) * 8
    new_height = math.ceil(height / 8) * 8
    return new_width, new_height

def scale_image_to_max_dimension(image, max_dimension=1024):
    """Scale image to have max dimension while maintaining aspect ratio."""
    ratio = max_dimension / max(image.width, image.height)
    if ratio < 1:  # Only scale down, not up
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        return new_width, new_height
    return image.width, image.height

# Load the inpainting pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

# Define input paths
input_image_path = 'data/page_20.jpeg'
input_mask_path = 'data/page_20_sam2_mask_4.png' 

# Define prompt
prompt = "text of medieval manuscript, respecting page margins and column layout on all sides, maintaining consistent text height with surrounding columns, following the same vertical alignment as existing text"

try:
    # Load the images
    init_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(input_mask_path).convert("RGB")
    
    # First scale down if needed
    new_width, new_height = scale_image_to_max_dimension(init_image, max_dimension=1024)
    print(f"Scaling down to: {new_width}x{new_height}")
    
    # Then ensure dimensions are divisible by 8
    new_width, new_height = get_dimensions_divisible_by_8(new_width, new_height)
    print(f"Final dimensions: {new_width}x{new_height}")
    
    # Resize both images
    init_image = init_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    mask_image = mask_image.resize((new_width, new_height), Image.Resampling.NEAREST)
    
    # Generate the inpainting
    result = pipeline(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        height=new_height,
        width=new_width,
        num_inference_steps=50,    # Increased for better quality
        guidance_scale=7.5,       # Increased to be more strict about following layout
        negative_prompt="text extending beyond margins, text not aligned with columns, inconsistent text height, text going past the bottom margin",  # Added negative prompt
    )
    inpainted_image = result.images[0]
    
    # Save the inpainting result
    output_path = f"data/{input_image_path.split('/')[-1].replace('.jpeg', '')}_inpainted.png"
    inpainted_image.save(output_path)
    print(f"Inpainted image saved to {output_path}")
    
    # Create a comparison image
    comparison_image = Image.new('RGB', (new_width * 2, new_height))
    comparison_image.paste(init_image, (0, 0))
    comparison_image.paste(inpainted_image, (new_width, 0))
    
    # Save the comparison image
    comparison_output_path = f"data/{input_image_path.split('/')[-1].replace('.jpeg', '')}_comparison.png"
    comparison_image.save(comparison_output_path)
    print(f"Comparison image saved to {comparison_output_path}")

except Exception as e:
    print(f"Error during processing: {e}")