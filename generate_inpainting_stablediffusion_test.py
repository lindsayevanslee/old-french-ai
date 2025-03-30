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
    "runwayml/stable-diffusion-inpainting",  # Back to the original inpainting model
    torch_dtype=torch.float16,
)
pipeline = pipeline.to("cuda")

# Define input paths
# input_image_path = 'data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/page_20.jpeg'
# input_mask_path = 'data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/page_20_sam2_mask_4.png' 

# input_image_path = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/page_37.jpeg'
# input_mask_path = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/page_37_sam2_mask_2.png' 

input_image_path = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/page_120.jpeg'
input_mask_path = 'data/digitized versions/Manuscrits numerises de la Bibliotheque municipale de Toulouse/page_120_sam2_mask_7.png' 


# Define prompt
# prompt = """medieval manuscript text, two-column layout, precise margins, 
# text aligned with existing columns, exact same text height as surrounding text, 
# respecting bottom margin, maintaining consistent line spacing, 
# text stopping at the same height as the second column"""

prompt = """medieval illumination style figure drawing, three distinct human figures in brown Benedictine habits,
monks seated at wooden table studying manuscript, clear faces with tonsured heads and beards,
detailed medieval clothing with flowing robes and hoods, figures interacting with the book,
painted in flat medieval art style with strong outlines, muted earth tones and rich jewel colors,
composition similar to medieval manuscript illustrations, figures arranged in triangular composition,
architectural details of stone arches and columns in background"""



# Define negative prompt
# negative_prompt = """text extending past bottom margin, text going beyond column boundaries,
# text not aligned with existing columns, inconsistent line height, 
# text continuing past the last line of the second column,
# text extending beyond the page margins, text not respecting layout"""

negative_prompt = """abstract shapes, decorative patterns without figures, incomplete human forms,
modern art style, photorealism, digital art effects, anime or cartoon style,
missing faces or bodies, floating elements, unclear figures,
modern clothing or accessories, contemporary setting,
oversaturated colors, blurry details, missing architectural elements"""


try:
    # Load the images
    init_image = Image.open(input_image_path).convert("RGB")
    mask_image = Image.open(input_mask_path).convert("RGB")
    
    # Get the directory path from input image
    output_dir = os.path.dirname(input_image_path)
    base_filename = os.path.splitext(os.path.basename(input_image_path))[0]
    
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
        guidance_scale=8.5,        # Moderate guidance scale
        negative_prompt=negative_prompt,
    )
    inpainted_image = result.images[0]
    
    # Save the inpainting result
    output_path = os.path.join(output_dir, f"{base_filename}_inpainted.png")
    inpainted_image.save(output_path)
    print(f"Inpainted image saved to {output_path}")
    
    # Create a comparison image
    comparison_image = Image.new('RGB', (new_width * 2, new_height))
    comparison_image.paste(init_image, (0, 0))
    comparison_image.paste(inpainted_image, (new_width, 0))
    
    # Save the comparison image
    comparison_output_path = os.path.join(output_dir, f"{base_filename}_comparison.png")
    comparison_image.save(comparison_output_path)
    print(f"Comparison image saved to {comparison_output_path}")

except Exception as e:
    print(f"Error during processing: {e}")