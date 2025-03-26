import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

def generate_damage_mask(image_path, masks_dir, overlays_dir, debug_dir=None):
    """
    Generate a binary mask highlighting damaged areas in a manuscript image.
    
    Args:
        image_path: Path to the input image
        masks_dir: Directory to save the output mask
        overlays_dir: Directory to save the overlay image
        debug_dir: Directory to save debug images (None to disable debug output)
        
    Returns:
        Tuple of (mask_path, overlay_path)
    """
    # Create output directories if they don't exist
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Convert paths to Path objects
    image_path = Path(image_path)
    masks_dir = Path(masks_dir)
    overlays_dir = Path(overlays_dir)
    
    # Only convert debug_dir to Path if it's not None
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
    
    # Read the input image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Extract the filename without extension
    filename = image_path.stem
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "2_grayscale.jpg"), gray)
    
    # Step 2: Apply adaptive thresholding to identify text regions
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "3_adaptive_thresh.jpg"), adaptive_thresh)
    
    # Step 3: Find contours to identify text regions
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for text regions
    content_mask = np.zeros_like(gray)
    min_area = 500  # Minimum area for text regions
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(content_mask, [contour], -1, 255, -1)
    
    if debug_dir:
        cv2.imwrite(str(debug_dir / "4_content_mask.jpg"), content_mask)
    
    # Step 4: Calculate local density
    kernel_size = 25
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    density = cv2.filter2D(gray, -1, kernel / (kernel_size * kernel_size))
    if debug_dir:
        cv2.imwrite(str(debug_dir / "5_density.jpg"), density)
    
    # Threshold density to identify areas with unusual density
    _, density_binary = cv2.threshold(density, 200, 255, cv2.THRESH_BINARY_INV)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "6_density_binary.jpg"), density_binary)
    
    # Step 5: Calculate local contrast
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_contrast = cv2.absdiff(gray, blur)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "7_local_contrast.jpg"), local_contrast)
    
    # Threshold local contrast to identify areas with unusual texture
    _, contrast_binary = cv2.threshold(local_contrast, 15, 255, cv2.THRESH_BINARY)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "8_contrast_binary.jpg"), contrast_binary)
    
    # Step 6: Combine density and contrast masks
    combined_mask = cv2.bitwise_or(density_binary, contrast_binary)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "9_combined_mask.jpg"), combined_mask)
    
    # Step 7: Analyze color (saturation)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Threshold saturation to identify areas with low saturation (often damaged)
    _, saturation_binary = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY_INV)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "10_saturation_binary.jpg"), saturation_binary)
    
    # Step 8: Create initial damage mask
    initial_damage_mask = cv2.bitwise_and(combined_mask, saturation_binary)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "11_initial_damage_mask.jpg"), initial_damage_mask)
    
    # Step 9: Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    morphology = cv2.morphologyEx(initial_damage_mask, cv2.MORPH_CLOSE, kernel)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_OPEN, kernel)
    if debug_dir:
        cv2.imwrite(str(debug_dir / "12_morphology.jpg"), morphology)
    
    # Step 10: Filter out small regions
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphology, connectivity=8)
    
    # Filter out small components
    min_damage_area = 100
    filtered_mask = np.zeros_like(morphology)
    
    # Start from 1 to skip the background label
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_damage_area:
            filtered_mask[labels == i] = 255
    
    if debug_dir:
        cv2.imwrite(str(debug_dir / "13_filtered_mask.jpg"), filtered_mask)
    
    # Step 11: Add intensity-based damage detection
    # Areas that are significantly darker or lighter than the average
    intensity_threshold = 50
    avg_intensity = np.mean(gray)
    
    intensity_mask = np.zeros_like(gray)
    intensity_mask[(gray < avg_intensity - intensity_threshold) | 
                  (gray > avg_intensity + intensity_threshold)] = 255
    
    if debug_dir:
        cv2.imwrite(str(debug_dir / "14_intensity_mask.jpg"), intensity_mask)
    
    # Step 12: Combine with the filtered mask
    final_mask = cv2.bitwise_or(filtered_mask, intensity_mask)
    
    # Save the original image for debug
    if debug_dir:
        cv2.imwrite(str(debug_dir / "1_original.jpg"), image)
        cv2.imwrite(str(debug_dir / "15_final_mask.jpg"), final_mask)
    
    # Create an overlay image
    overlay = image.copy()
    overlay[final_mask > 0] = [0, 0, 255]  # Red color for damaged areas
    
    # Save the mask and overlay
    mask_path = masks_dir / f"{filename}_mask.jpg"
    overlay_path = overlays_dir / f"{filename}_overlay.jpg"
    
    cv2.imwrite(str(mask_path), final_mask)
    cv2.imwrite(str(overlay_path), overlay)
    
    return str(mask_path), str(overlay_path)

def process_directory(input_dir, output_parent_dir=None):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_parent_dir: Parent directory for output folders (if None, use parent of input_dir)
    """
    # Convert to Path objects
    input_dir = Path(input_dir)
    
    # If no output parent directory is specified, use the parent of the input directory
    if output_parent_dir is None:
        output_parent_dir = input_dir.parent
    else:
        output_parent_dir = Path(output_parent_dir)
    
    # Set up output directories
    mask_dir = output_parent_dir / "masks"
    overlay_dir = output_parent_dir / "overlays"
    debug_dir = output_parent_dir / "debug"
    
    # Create output directories
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images in the input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(input_dir.glob(ext)))
        image_paths.extend(list(input_dir.glob(f"**/{ext}")))
    
    # Remove duplicates and sort
    image_paths = sorted(list(set(image_paths)))
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Output directories:")
    print(f"  Masks: {mask_dir}")
    print(f"  Overlays: {overlay_dir}")
    print(f"  Debug: {debug_dir}")
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Generating damage masks"):
        try:
            # Create debug directory for this image
            img_debug_dir = debug_dir / image_path.stem
            img_debug_dir.mkdir(parents=True, exist_ok=True)
            
            mask_path, overlay_path = generate_damage_mask(
                image_path, 
                mask_dir, 
                overlay_dir, 
                img_debug_dir
            )
            
            if mask_path is None:
                print(f"Failed to process {image_path}")
            else:
                relative_path = image_path.relative_to(input_dir) if image_path.is_relative_to(input_dir) else image_path.name
                print(f"Generated mask for {relative_path}")
                print(f"  Mask: {mask_path}")
                print(f"  Overlay: {overlay_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

def process_single_image(image_path, output_parent_dir=None):
    """
    Process a single image
    
    Args:
        image_path: Path to the input image
        output_parent_dir: Parent directory for output folders (if None, use parent directory of image)
    """
    # Convert to Path objects
    image_path = Path(image_path)
    
    # If no output parent directory is specified, use the parent directory of the image
    if output_parent_dir is None:
        output_parent_dir = image_path.parent.parent
    else:
        output_parent_dir = Path(output_parent_dir)
    
    # Set up output directories
    mask_dir = output_parent_dir / "masks"
    overlay_dir = output_parent_dir / "overlays"
    debug_dir = output_parent_dir / "debug" / image_path.stem
    
    # Create output directories
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directories:")
    print(f"  Masks: {mask_dir}")
    print(f"  Overlays: {overlay_dir}")
    print(f"  Debug: {debug_dir}")
    
    try:
        mask_path, overlay_path = generate_damage_mask(
            image_path, 
            mask_dir, 
            overlay_dir, 
            debug_dir
        )
        
        if mask_path is None:
            print(f"Failed to process {image_path}")
        else:
            print(f"Generated mask for {image_path.name}")
            print(f"  Mask: {mask_path}")
            print(f"  Overlay: {overlay_path}")
            print(f"  Debug images: {debug_dir}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate damage masks for manuscript pages")
    parser.add_argument("--input", "-i", help="Input directory containing images")
    parser.add_argument("--output", "-o", help="Parent directory for output folders (if not specified, uses parent of input directory)")
    parser.add_argument("--single", "-s", help="Process a single image instead of a directory")
    
    args = parser.parse_args()
    
    if args.single:
        if not os.path.isfile(args.single):
            print(f"Error: {args.single} is not a file")
            return
        
        print(f"Processing single image: {args.single}")
        process_single_image(args.single, args.output)
    else:
        # If no input directory is specified, use the same one as generate_masks.py
        input_dir = args.input
        if not input_dir:
            default_input_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/jpeg"
            print(f"No input directory specified, using default: {default_input_dir}")
            input_dir = default_input_dir
            
        if not os.path.isdir(input_dir):
            print(f"Error: {input_dir} is not a directory")
            return
        
        print(f"Processing directory: {input_dir}")
        process_directory(input_dir, args.output)

if __name__ == "__main__":
    main() 