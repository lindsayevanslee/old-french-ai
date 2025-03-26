#!/usr/bin/env python3
# combined_damage_detection.py
# A script that combines the best aspects of both the original and RGB-based approaches

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib.util
import time

#output prints to data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/mask_methods/combined_approach

# Import functions from both approaches
from generate_masks_rgb import detect_damage_rgb
from generate_masks_damage import generate_damage_mask

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def combined_damage_detection(image_path, output_dir="data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/mask_methods/combined_approach", debug=True):
    """
    Combine the original and RGB-based approaches for damage detection
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
        debug: Whether to save debug images
        
    Returns:
        Tuple of (mask_path, overlay_path)
    """
    # Extract filename without extension for output naming
    image_filename = Path(image_path).stem
    
    # Create output directories
    masks_dir = os.path.join(output_dir, 'masks')
    overlays_dir = os.path.join(output_dir, 'overlays')
    debug_dir = os.path.join(output_dir, 'debug', image_filename)
    
    create_directory(masks_dir)
    create_directory(overlays_dir)
    
    if debug:
        create_directory(debug_dir)
    
    # Create temporary directories for individual approaches
    temp_original_dir = os.path.join(output_dir, 'temp_original')
    temp_rgb_dir = os.path.join(output_dir, 'temp_rgb')
    
    create_directory(temp_original_dir)
    create_directory(temp_rgb_dir)
    
    # Run the original approach
    print(f"Running original approach on: {image_path}")
    original_masks_dir = os.path.join(temp_original_dir, 'masks')
    original_overlays_dir = os.path.join(temp_original_dir, 'overlays')
    original_debug_dir = os.path.join(temp_original_dir, 'debug', image_filename)
    
    create_directory(original_masks_dir)
    create_directory(original_overlays_dir)
    create_directory(original_debug_dir)
    
    original_mask_path, original_overlay_path = generate_damage_mask(
        image_path, 
        original_masks_dir, 
        original_overlays_dir, 
        original_debug_dir
    )
    
    # Run the RGB-based approach
    print(f"Running RGB-based approach on: {image_path}")
    rgb_mask_path, rgb_overlay_path = detect_damage_rgb(image_path, temp_rgb_dir)
    
    # Read the original image
    original_image = cv2.imread(image_path)
    
    # Read the masks from both approaches
    original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
    rgb_mask = cv2.imread(rgb_mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert masks to binary (0 or 255)
    _, original_mask_binary = cv2.threshold(original_mask, 127, 255, cv2.THRESH_BINARY)
    _, rgb_mask_binary = cv2.threshold(rgb_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create a combined mask using the strengths of both approaches
    # 1. Start with the union of both masks
    combined_mask = cv2.bitwise_or(original_mask_binary, rgb_mask_binary)
    
    # 2. Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Remove small isolated regions
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
    
    # Filter out small components (adjust min_size as needed)
    min_size = 50
    filtered_mask = np.zeros_like(combined_mask)
    
    # Start from 1 to skip the background label
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_mask[labels == i] = 255
    
    # 4. Create a refined mask that focuses on areas where both approaches agree
    # but also includes unique detections from each approach
    agreement_mask = cv2.bitwise_and(original_mask_binary, rgb_mask_binary)
    
    # Get unique detections from each approach
    original_unique = cv2.bitwise_and(original_mask_binary, cv2.bitwise_not(rgb_mask_binary))
    rgb_unique = cv2.bitwise_and(rgb_mask_binary, cv2.bitwise_not(original_mask_binary))
    
    # Weight the unique detections (can be adjusted)
    original_weight = 0.7  # Higher weight for original approach's unique detections
    rgb_weight = 0.9       # Higher weight for RGB approach's unique detections
    
    # Create a weighted mask
    weighted_mask = np.zeros_like(combined_mask, dtype=np.float32)
    weighted_mask[agreement_mask > 0] = 1.0  # Areas where both agree get full weight
    weighted_mask[original_unique > 0] = original_weight
    weighted_mask[rgb_unique > 0] = rgb_weight
    
    # Threshold the weighted mask to get the final mask
    _, refined_mask = cv2.threshold((weighted_mask * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    
    # Apply the filtered mask to remove small regions
    final_mask = cv2.bitwise_and(refined_mask, filtered_mask)
    
    # Create an overlay image
    overlay = original_image.copy()
    overlay[final_mask > 0] = [0, 0, 255]  # Red color for damaged areas
    
    # Save the final mask and overlay
    final_mask_path = os.path.join(masks_dir, f"{image_filename}_mask.jpg")
    final_overlay_path = os.path.join(overlays_dir, f"{image_filename}_overlay.jpg")
    
    cv2.imwrite(final_mask_path, final_mask)
    cv2.imwrite(final_overlay_path, overlay)
    
    print(f"Saved final mask: {final_mask_path}")
    print(f"Saved overlay: {final_overlay_path}")
    
    # Save debug images if requested
    if debug:
        # Save the original and RGB masks
        cv2.imwrite(os.path.join(debug_dir, "01_original_mask.jpg"), original_mask)
        cv2.imwrite(os.path.join(debug_dir, "02_rgb_mask.jpg"), rgb_mask)
        
        # Save the combined mask
        cv2.imwrite(os.path.join(debug_dir, "03_combined_mask.jpg"), combined_mask)
        
        # Save the filtered mask
        cv2.imwrite(os.path.join(debug_dir, "04_filtered_mask.jpg"), filtered_mask)
        
        # Save the agreement mask
        cv2.imwrite(os.path.join(debug_dir, "05_agreement_mask.jpg"), agreement_mask)
        
        # Save the unique detections
        cv2.imwrite(os.path.join(debug_dir, "06_original_unique.jpg"), original_unique)
        cv2.imwrite(os.path.join(debug_dir, "07_rgb_unique.jpg"), rgb_unique)
        
        # Save the weighted mask
        cv2.imwrite(os.path.join(debug_dir, "08_weighted_mask.jpg"), (weighted_mask * 255).astype(np.uint8))
        
        # Save the refined mask
        cv2.imwrite(os.path.join(debug_dir, "09_refined_mask.jpg"), refined_mask)
        
        # Save the final mask
        cv2.imwrite(os.path.join(debug_dir, "10_final_mask.jpg"), final_mask)
        
        # Create a color-coded comparison image
        comparison = np.zeros((original_mask.shape[0], original_mask.shape[1], 3), dtype=np.uint8)
        
        # Red channel: Areas detected only by the original approach
        comparison[:,:,2] = original_unique
        
        # Green channel: Areas detected only by the RGB approach
        comparison[:,:,1] = rgb_unique
        
        # Blue channel: Areas detected by both approaches
        comparison[:,:,0] = agreement_mask
        
        # Save the comparison image
        cv2.imwrite(os.path.join(debug_dir, "11_comparison.jpg"), comparison)
        
        # Create a side-by-side comparison
        h, w = original_image.shape[:2]
        scale = 0.5  # Resize for better visualization
        
        original_image_resized = cv2.resize(original_image, (0, 0), fx=scale, fy=scale)
        original_mask_resized = cv2.resize(original_mask, (0, 0), fx=scale, fy=scale)
        rgb_mask_resized = cv2.resize(rgb_mask, (0, 0), fx=scale, fy=scale)
        final_mask_resized = cv2.resize(final_mask, (0, 0), fx=scale, fy=scale)
        
        h_resized, w_resized = original_image_resized.shape[:2]
        side_by_side = np.zeros((h_resized, w_resized * 4, 3), dtype=np.uint8)
        
        # Original image
        side_by_side[:, 0:w_resized] = original_image_resized
        
        # Original mask (convert to 3-channel)
        original_mask_color = cv2.cvtColor(original_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized:w_resized*2] = original_mask_color
        
        # RGB mask (convert to 3-channel)
        rgb_mask_color = cv2.cvtColor(rgb_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized*2:w_resized*3] = rgb_mask_color
        
        # Final mask (convert to 3-channel)
        final_mask_color = cv2.cvtColor(final_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized*3:w_resized*4] = final_mask_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Original Approach", (w_resized + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "RGB Approach", (w_resized*2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Combined Approach", (w_resized*3 + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Save the side-by-side comparison
        cv2.imwrite(os.path.join(debug_dir, "12_side_by_side.jpg"), side_by_side)
    
    return final_mask_path, final_overlay_path

def process_directory(input_dir, output_dir="data/combined_approach", debug=True):
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        debug: Whether to save debug images
    """
    # Create output directories
    create_directory(output_dir)
    
    # Get all JPEG files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpeg', '.jpg'))]
    total_images = len(image_files)
    
    print(f"Found {total_images} images in {input_dir}")
    
    # Process each image
    start_time = time.time()
    
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(input_dir, image_file)
        
        # Calculate progress
        progress = (i + 1) / total_images * 100
        elapsed_time = time.time() - start_time
        images_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"Processing image {i+1}/{total_images} ({progress:.2f}%): {image_file}")
        
        try:
            combined_damage_detection(image_path, output_dir, debug)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
        
        # Print progress
        if (i + 1) % 10 == 0 or (i + 1) == total_images:
            print(f"Progress: {i+1}/{total_images} images ({progress:.2f}%) - "
                  f"{images_per_second:.2f} images/second")
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nProcessing completed!")
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {total_images / total_time:.2f} images/second")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined damage detection for manuscript images")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/mask_methods/combined_approach", 
                        help="Directory to save output images")
    parser.add_argument("--single", type=str, help="Process a single image")
    parser.add_argument("--no_debug", action="store_true", help="Disable debug images")
    
    args = parser.parse_args()
    
    # Process a single image or a directory
    if args.single:
        if os.path.isfile(args.single):
            combined_damage_detection(args.single, args.output_dir, not args.no_debug)
        else:
            print(f"Error: {args.single} is not a file")
    elif args.input_dir:
        if os.path.isdir(args.input_dir):
            process_directory(args.input_dir, args.output_dir, not args.no_debug)
        else:
            print(f"Error: {args.input_dir} is not a directory")
    else:
        # Default input directory
        default_input_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/jpeg"
        if os.path.isdir(default_input_dir):
            process_directory(default_input_dir, args.output_dir, not args.no_debug)
        else:
            print(f"Error: Default input directory {default_input_dir} not found")
            print("Please specify an input directory with --input_dir or a single image with --single")

if __name__ == "__main__":
    main() 