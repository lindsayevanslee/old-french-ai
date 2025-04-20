"""
Generate inpainting using Stable Diffusion model without pretraining. Run on one image at a time.
"""
import os
import json
import argparse
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import math

# Default configuration name - change this to use a different configuration
DEFAULT_CONFIG_NAME = "toulouse_page120"

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

def load_config(config_path, config_name):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    for conf in config['model_inputs']:
        if conf['name'] == config_name:
            return conf
    
    raise ValueError(f"Configuration '{config_name}' not found in {config_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate inpainting using Stable Diffusion')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    parser.add_argument('--config-name', type=str, default=DEFAULT_CONFIG_NAME, 
                        help=f'Name of the configuration to use (default: {DEFAULT_CONFIG_NAME})')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config, args.config_name)
    
    # Load the inpainting pipeline
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
    )
    pipeline = pipeline.to("cuda")

    try:
        # Load the images
        init_image = Image.open(config['input_image_path']).convert("RGB")
        mask_image = Image.open(config['input_mask_path']).convert("RGB")
        
        # Get the directory path from input image
        output_dir = f"{os.path.dirname(os.path.dirname(config['input_image_path']))}/model_results/stablediffusion_oob"
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(config['input_image_path']))[0]
        
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
            prompt=config['prompt'],
            image=init_image,
            mask_image=mask_image,
            height=new_height,
            width=new_width,
            num_inference_steps=50,    # Increased for better quality
            guidance_scale=8.5,        # Moderate guidance scale
            negative_prompt=config['negative_prompt'],
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

if __name__ == "__main__":
    main()