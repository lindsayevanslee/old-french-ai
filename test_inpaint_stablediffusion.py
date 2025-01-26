import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from torchvision import transforms

def preprocess_images(image_path, mask_path, target_size=512):
    """Preprocess images for the inpainting pipeline with validation checks."""
    # Load and resize the original image
    image = Image.open(image_path).convert('RGB')
    
    # Load mask and convert to binary black and white
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale first
    
    # Resize both to target size
    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size)
    ])
    
    image = resize_transform(image)
    mask = resize_transform(mask)
    
    # Convert mask to binary (0 or 255)
    mask_array = np.array(mask)
    mask_array = (mask_array > 127.5).astype(np.uint8) * 255
    mask = Image.fromarray(mask_array, mode='L')
    
    # Print validation information
    print(f"Image size: {image.size}")
    print(f"Mask size: {mask.size}")
    print(f"Unique mask values: {np.unique(mask_array)}")
    
    return image, mask

def main():
    # Path to your newly saved inpainting model
    model_dir = "manuscript_restoration_model2"
    
    # Load the pipeline with proper configuration
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Ensure the model is in evaluation mode
    pipe.unet.eval()

    # Check for NaN weights
    for name, param in pipe.unet.named_parameters():
        if torch.isnan(param).any():
            print(f"Found NaN weights in {name}")
    
    # Define paths
    test_mutilated_path = "data/digitized versions/Vies des saints/mutilations/page_11_mutilated.jpeg"
    test_mask_path = "data/digitized versions/Vies des saints/masks/page_11_mask.jpeg"
    
    # Preprocess images
    image, mask = preprocess_images(test_mutilated_path, test_mask_path)
    
    # Set inference parameters
    prompt = "restore medieval manuscript text, high quality, detailed"
    num_inference_steps = 50  # Increase for better quality
    guidance_scale = 7.5     # Controls how much the image follows the prompt
    
    # Perform the inpainting with error handling
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        # Save the result
        output_path = "data/digitized versions/Vies des saints/model_results/stablediffusion_trained/restored_page.png"
        result.save(output_path)
        print(f"Successfully saved restored image to {output_path}")
        
        # Also save the preprocessed inputs for verification
        image.save(output_path.replace('.png', '_input.png'))
        mask.save(output_path.replace('.png', '_mask.png'))
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()