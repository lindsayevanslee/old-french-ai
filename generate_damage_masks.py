import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from pathlib import Path

def generate_damage_mask(image_path, mask_dir, overlay_dir, debug_dir):
    """
    Generate a damage mask for a manuscript page
    
    Args:
        image_path: Path to the input image
        mask_dir: Directory to save the output mask
        overlay_dir: Directory to save the overlay visualization
        debug_dir: Directory to save debug images
    
    Returns:
        Tuple of (mask_path, overlay_path)
    """
    # Convert paths to Path objects if they're strings
    image_path = Path(image_path)
    mask_dir = Path(mask_dir)
    overlay_dir = Path(overlay_dir)
    debug_dir = Path(debug_dir)
    
    # Create output directories
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the image
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Get dimensions
    height, width = gray.shape
    
    # Save debug images
    cv2.imwrite(str(debug_dir / "1_original.jpg"), original)
    cv2.imwrite(str(debug_dir / "2_grayscale.jpg"), gray)
    
    # 1. Apply adaptive thresholding to identify text regions
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(str(debug_dir / "3_adaptive_thresh.jpg"), adaptive_thresh)
    
    # 2. Find text regions using contours
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the content area
    content_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Filter contours by area and draw the largest ones
    min_area = 500  # Minimum area to be considered a text region
    text_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            text_regions.append((x, y, w, h, area))
    
    # Sort by area (largest first)
    text_regions.sort(key=lambda x: x[4], reverse=True)
    
    # Draw all text regions to create a comprehensive content mask
    for x, y, w, h, area in text_regions:
        cv2.rectangle(content_mask, (x, y), (x+w, y+h), 255, -1)
    
    # Dilate the content mask to include surrounding areas
    kernel = np.ones((20, 20), np.uint8)
    content_mask = cv2.dilate(content_mask, kernel, iterations=1)
    cv2.imwrite(str(debug_dir / "4_content_mask.jpg"), content_mask)
    
    # 3. Apply local density analysis with a smaller kernel
    kernel_size = 25  # Smaller kernel to capture more local details
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    density = cv2.filter2D(gray, -1, kernel)
    
    # Normalize density for visualization
    density_normalized = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(debug_dir / "5_density.jpg"), density_normalized)
    
    # 4. Create binary mask for low density areas (potential damage)
    # Use a threshold based on the mean density of the image
    density_threshold = np.mean(density) * 0.95  # Slightly below mean
    
    # Create binary mask for pixels with density below threshold
    density_binary = np.zeros_like(gray, dtype=np.uint8)
    density_binary[density < density_threshold] = 255
    cv2.imwrite(str(debug_dir / "6_density_binary.jpg"), density_binary)
    
    # 5. Analyze local contrast (texture)
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_contrast = cv2.absdiff(gray, blur)
    
    # Normalize for visualization
    contrast_normalized = cv2.normalize(local_contrast, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(debug_dir / "7_local_contrast.jpg"), contrast_normalized)
    
    # Find contrast threshold based on image statistics
    contrast_threshold = np.mean(local_contrast) * 0.5  # Half of mean contrast
    
    # Create binary mask for low contrast areas (potential damage)
    contrast_binary = np.zeros_like(gray, dtype=np.uint8)
    contrast_binary[local_contrast < contrast_threshold] = 255
    cv2.imwrite(str(debug_dir / "8_contrast_binary.jpg"), contrast_binary)
    
    # 6. Combine density and contrast masks
    combined_mask = cv2.bitwise_and(density_binary, contrast_binary)
    cv2.imwrite(str(debug_dir / "9_combined_mask.jpg"), combined_mask)
    
    # 7. Apply color analysis in HSV space for additional features
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Analyze saturation
    saturation_threshold = np.mean(s) * 0.8  # 80% of mean saturation
    
    # Create binary mask for areas with low saturation
    saturation_binary = np.zeros_like(gray, dtype=np.uint8)
    saturation_binary[s < saturation_threshold] = 255
    cv2.imwrite(str(debug_dir / "10_saturation_binary.jpg"), saturation_binary)
    
    # 8. Combine with content mask and all other features
    initial_damage_mask = cv2.bitwise_and(combined_mask, content_mask)
    initial_damage_mask = cv2.bitwise_and(initial_damage_mask, saturation_binary)
    cv2.imwrite(str(debug_dir / "11_initial_damage_mask.jpg"), initial_damage_mask)
    
    # 9. Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    damage_mask = cv2.morphologyEx(initial_damage_mask, cv2.MORPH_OPEN, kernel)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(str(debug_dir / "12_morphology.jpg"), damage_mask)
    
    # 10. Filter small regions
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(damage_mask)
    
    min_damage_area = 100  # Minimum area to be considered damage
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_damage_area:
            cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    cv2.imwrite(str(debug_dir / "13_filtered_mask.jpg"), filtered_mask)
    
    # 11. Apply intensity-based filtering
    intensity_threshold = np.mean(gray) * 0.95  # 95% of mean intensity
    
    # Create a mask for areas with intensity below threshold
    intensity_mask = np.zeros_like(gray, dtype=np.uint8)
    intensity_mask[gray < intensity_threshold] = 255
    cv2.imwrite(str(debug_dir / "14_intensity_mask.jpg"), intensity_mask)
    
    # 12. Combine filtered mask with intensity mask
    final_mask = cv2.bitwise_or(filtered_mask, cv2.bitwise_and(intensity_mask, content_mask))
    cv2.imwrite(str(debug_dir / "15_final_mask.jpg"), final_mask)
    
    # 13. Apply a final cleanup with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Create overlay visualization
    overlay = original.copy()
    # Add red tint to damaged areas
    overlay[final_mask > 0] = (0, 0, 255)  # Red color for damaged areas
    # Blend with original image
    alpha = 0.5
    overlay = cv2.addWeighted(original, 1-alpha, overlay, alpha, 0)
    
    # Save the final mask and overlay
    mask_path = mask_dir / f"{image_path.stem}_mask.jpg"
    overlay_path = overlay_dir / f"{image_path.stem}_overlay.jpg"
    
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