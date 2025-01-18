"""
Apply inpainting from trained model using pre-computed masks and create seamless overlays.
This script loads a trained inpainting model and generates results using pre-computed masks,
creating both comparison visualizations and realistic overlays of the inpainted regions.
"""

import torch
from inpainting_model import UNetInpaint
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import torch.nn.functional as F

def load_and_process_image(mutilated_path, mask_path, img_size=1000):
    """
    Load and process images for inpainting using pre-computed masks.
    
    Args:
        mutilated_path (str): Path to the mutilated image
        mask_path (str): Path to the pre-computed mask
        img_size (int): Size to resize images to (default: 1000 to match training)
        
    Returns:
        tuple: (input_tensor, original_size, original_image)
            - input_tensor: Processed input tensor ready for the model
            - original_size: Original image dimensions (height, width)
            - original_image: Original mutilated image as PIL Image
    """
    # Load images
    original_image = Image.open(mutilated_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Load mask as grayscale
    
    # Store original size
    original_size = original_image.size
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Apply transforms
    mutilated = transform(original_image)
    mask = transform(mask)
    
    # Ensure mask is binary
    mask_binary = (mask > 0.5).float()
    
    # Combine image and mask
    input_tensor = torch.cat([mutilated, mask_binary], dim=0)
    return input_tensor.unsqueeze(0), original_size, original_image

def create_inpainted_overlay(output_image, original_image, mask_image, feather_amount=3):
    """
    Create a seamless overlay of the inpainted region on the original image.
    
    Args:
        output_image (numpy.ndarray): The model's output image
        original_image (PIL.Image): The original mutilated image
        mask_image (PIL.Image): The binary mask image
        feather_amount (int): Amount of feathering for the mask edges
        
    Returns:
        numpy.ndarray: The final overlaid image
    """
    # Convert everything to numpy arrays in the same size
    original_array = np.array(original_image)
    mask_array = np.array(mask_image.resize(original_image.size, Image.Resampling.LANCZOS)).astype(float) / 255.0
    
    # Resize output image to match original size
    output_resized = np.array(Image.fromarray((output_image * 255).astype(np.uint8)).resize(
        original_image.size, Image.Resampling.LANCZOS))
    output_resized = output_resized.astype(float) / 255.0
    
    # Create feathered mask
    from scipy.ndimage import gaussian_filter
    feathered_mask = gaussian_filter(mask_array, sigma=feather_amount)
    
    # Expand dimensions for broadcasting
    feathered_mask = feathered_mask[..., np.newaxis]
    
    # Blend images using the feathered mask
    blended = (output_resized * feathered_mask) + (original_array.astype(float) / 255.0 * (1 - feathered_mask))
    
    return (blended * 255).astype(np.uint8)

def generate_inpainting(mutilated_path, excised_path, mask_path):
    """
    Generate and visualize inpainting results with overlay.
    
    Args:
        mutilated_path (str): Path to the mutilated image
        excised_path (str): Path to the excised content (for visualization)
        mask_path (str): Path to the pre-computed mask
    """
    # Extract page number from the path
    page_number = Path(mutilated_path).stem.split('_')[1]
    
    # Load the trained model
    model = UNetInpaint()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/unet_inpaint_best.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # Load and process images
    input_tensor, original_size, original_image = load_and_process_image(mutilated_path, mask_path)
    input_tensor = input_tensor.to(device)
    
    # Generate inpainting
    print("Generating inpainting...")
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output to image
    output_image = output.cpu().squeeze(0).permute(1, 2, 0).numpy()
    output_image = output_image.clip(0, 1)  # Ensure values are in valid range
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Inpainting Results for Page {page_number}', fontsize=16)
    
    # Original mutilated image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Mutilated Image')
    axes[0, 0].axis('off')
    
    # Mask
    mask_img = Image.open(mask_path)
    axes[0, 1].imshow(mask_img, cmap='gray')
    axes[0, 1].set_title('Inpainting Mask')
    axes[0, 1].axis('off')
    
    # Excised content (ground truth)
    axes[1, 0].imshow(Image.open(excised_path))
    axes[1, 0].set_title('Excised Content (Ground Truth)')
    axes[1, 0].axis('off')
    
    # Inpainted result
    axes[1, 1].imshow(output_image)
    axes[1, 1].set_title('Inpainted Result')
    axes[1, 1].axis('off')
    
    # Save comparison figure
    save_dir = 'data/digitized versions/Vies des saints/model_results'
    os.makedirs(save_dir, exist_ok=True)
    comparison_path = os.path.join(save_dir, f'page_{page_number}_inpainting_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison image to {comparison_path}")
    
    # Create and save the overlaid result
    print("Creating inpainting overlay...")
    overlaid_image = create_inpainted_overlay(output_image, original_image, mask_img)
    overlay_path = os.path.join(save_dir, f'page_{page_number}_inpainted.png')
    
    # Save the overlay
    Image.fromarray(overlaid_image).save(overlay_path, dpi=(300, 300))
    print(f"Saved overlaid inpainting to {overlay_path}")
    
    # Display the overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid_image)
    plt.axis('off')
    plt.title('Inpainting Overlay')
    plt.show()

def main():
    """Main function to run the inpainting generation."""
    # Define paths
    base_dir = 'data/digitized versions/Vies des saints'
    mutilated_path = f'{base_dir}/mutilations/page_11_mutilated.jpeg'
    excised_path = f'{base_dir}/excisions/page_11_excised.jpeg'
    mask_path = f'{base_dir}/masks/page_11_mask.jpeg'
    
    # Verify all files exist
    for path in [mutilated_path, excised_path, mask_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    generate_inpainting(mutilated_path, excised_path, mask_path)

if __name__ == "__main__":
    main()