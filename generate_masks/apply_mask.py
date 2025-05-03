"""
Apply a mask to an image, creating a masked version where the masked areas are transparent.
The image will be cropped to remove excess black/transparent areas.
"""
import os
from PIL import Image
import numpy as np

def crop_transparent(image):
    """
    Crop the image to remove excess transparent/black areas.
    
    Args:
        image (PIL.Image): Input image with transparency
        
    Returns:
        PIL.Image: Cropped image
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Find non-transparent/black pixels
    if img_array.shape[2] == 4:  # RGBA
        mask = (img_array[:, :, 3] > 0) & (img_array[:, :, :3].any(axis=2) > 0)
    else:  # RGB
        mask = img_array.any(axis=2) > 0
    
    # Find the bounding box of non-transparent/black pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    # Crop the image
    return image.crop((left, top, right + 1, bottom + 1))

def apply_mask(image_path, mask_path, output_path=None):
    """
    Apply a mask to an image, creating a masked version with transparency.
    
    Args:
        image_path (str): Path to the input image
        mask_path (str): Path to the mask image (should be black and white)
        output_path (str, optional): Path to save the output. If None, will be derived from input path
    """
    # Load the images
    image = Image.open(image_path).convert('RGBA')  # Convert to RGBA for transparency
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    
    # Ensure images are the same size
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    
    # Convert to numpy arrays for processing
    image_array = np.array(image)
    mask_array = np.array(mask)
    
    # Create masked image (where mask is white (255), keep original image; where mask is black (0), make transparent)
    masked_array = image_array.copy()
    masked_array[mask_array == 0] = [0, 0, 0, 0]  # Set masked areas to transparent
    
    # Convert back to PIL Image
    masked_image = Image.fromarray(masked_array)
    
    # Crop the image to remove excess transparent areas
    cropped_image = crop_transparent(masked_image)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_masked.png"
    
    # Save the result
    cropped_image.save(output_path, 'PNG')
    print(f"Masked and cropped image saved to: {output_path}")

def main():
    # Hardcoded input paths
    image_path = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/presentation/page_20_inpainted.png"
    mask_path = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/presentation/page_20_sam2_mask_4.png"
    
    # Get the directory and base filename from the input image
    output_dir = os.path.dirname(image_path)
    base_filename = os.path.basename(image_path)
    # Extract the page number (e.g., "page_20" from "page_20_inpainted.png")
    page_number = base_filename.split('_')[0] + '_' + base_filename.split('_')[1]
    output_path = os.path.join(output_dir, f"{page_number}_masked.png")
    
    try:
        apply_mask(image_path, mask_path, output_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 