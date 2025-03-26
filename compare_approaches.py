#!/usr/bin/env python3
# compare_approaches.py
# A script to compare the RGB-based approach with the original approach

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib.util

# Import the RGB-based approach
from rgb_damage_detection import detect_damage_rgb

# Import the original approach
# We'll need to dynamically import the generate_damage_mask function from generate_damage_masks.py
def import_function(module_path, function_name):
    """Dynamically import a function from a module"""
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

# Import the original damage mask generation function
generate_damage_mask = import_function("generate_damage_masks.py", "generate_damage_mask")

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def compare_approaches(image_path, output_dir):
    """
    Compare the RGB-based approach with the original approach
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output and comparison images
    """
    # Extract filename without extension for output naming
    image_filename = Path(image_path).stem
    
    # Create output directories
    comparison_dir = os.path.join(output_dir, 'comparison', image_filename)
    create_directory(comparison_dir)
    
    # Create directories for the original approach
    original_masks_dir = os.path.join(output_dir, 'original', 'masks')
    original_overlays_dir = os.path.join(output_dir, 'original', 'overlays')
    original_debug_dir = os.path.join(output_dir, 'original', 'debug', image_filename)
    
    create_directory(original_masks_dir)
    create_directory(original_overlays_dir)
    create_directory(original_debug_dir)
    
    # Run the original approach
    print(f"Running original approach on: {image_path}")
    try:
        original_mask_path, original_overlay_path = generate_damage_mask(
            image_path, 
            original_masks_dir, 
            original_overlays_dir, 
            original_debug_dir
        )
        print(f"Original approach completed. Mask: {original_mask_path}, Overlay: {original_overlay_path}")
    except Exception as e:
        print(f"Error running original approach: {e}")
        original_mask_path = None
        original_overlay_path = None
    
    # Run the RGB-based approach
    print(f"Running RGB-based approach on: {image_path}")
    rgb_mask_path, rgb_overlay_path = detect_damage_rgb(image_path, output_dir)
    
    # If both approaches succeeded, create comparison visualizations
    if original_mask_path and rgb_mask_path:
        # Read the masks
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        rgb_mask = cv2.imread(rgb_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Create a comparison visualization
        # Convert masks to binary (0 or 255)
        _, original_mask_binary = cv2.threshold(original_mask, 127, 255, cv2.THRESH_BINARY)
        _, rgb_mask_binary = cv2.threshold(rgb_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Create a comparison image
        comparison = np.zeros((original_mask.shape[0], original_mask.shape[1], 3), dtype=np.uint8)
        
        # Red channel: Areas detected only by the original approach
        comparison[:,:,2] = cv2.bitwise_and(original_mask_binary, cv2.bitwise_not(rgb_mask_binary))
        
        # Green channel: Areas detected only by the RGB approach
        comparison[:,:,1] = cv2.bitwise_and(rgb_mask_binary, cv2.bitwise_not(original_mask_binary))
        
        # Blue channel: Areas detected by both approaches
        comparison[:,:,0] = cv2.bitwise_and(original_mask_binary, rgb_mask_binary)
        
        # Save the comparison image
        comparison_path = os.path.join(comparison_dir, f"{image_filename}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        print(f"Saved comparison image: {comparison_path}")
        
        # Calculate overlap statistics
        total_original = np.sum(original_mask_binary > 0)
        total_rgb = np.sum(rgb_mask_binary > 0)
        overlap = np.sum(cv2.bitwise_and(original_mask_binary, rgb_mask_binary) > 0)
        
        # Calculate Jaccard similarity (intersection over union)
        union = np.sum(cv2.bitwise_or(original_mask_binary, rgb_mask_binary) > 0)
        jaccard = overlap / union if union > 0 else 0
        
        # Calculate Dice coefficient (2 * intersection / sum of areas)
        dice = 2 * overlap / (total_original + total_rgb) if (total_original + total_rgb) > 0 else 0
        
        # Print statistics
        print(f"\nComparison Statistics for {image_filename}:")
        print(f"Original approach detected: {total_original} pixels")
        print(f"RGB approach detected: {total_rgb} pixels")
        print(f"Overlap (both approaches): {overlap} pixels")
        print(f"Jaccard similarity (IoU): {jaccard:.4f}")
        print(f"Dice coefficient: {dice:.4f}")
        
        # Create a side-by-side comparison of the original image, original mask, and RGB mask
        original_image = cv2.imread(image_path)
        
        # Resize for better visualization if needed
        scale = 0.5
        original_image_resized = cv2.resize(original_image, (0, 0), fx=scale, fy=scale)
        original_mask_resized = cv2.resize(original_mask, (0, 0), fx=scale, fy=scale)
        rgb_mask_resized = cv2.resize(rgb_mask, (0, 0), fx=scale, fy=scale)
        
        # Create a side-by-side comparison
        h, w = original_image_resized.shape[:2]
        side_by_side = np.zeros((h, w * 3, 3), dtype=np.uint8)
        
        # Original image
        side_by_side[:, 0:w] = original_image_resized
        
        # Original mask (convert to 3-channel)
        original_mask_color = cv2.cvtColor(original_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w:w*2] = original_mask_color
        
        # RGB mask (convert to 3-channel)
        rgb_mask_color = cv2.cvtColor(rgb_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w*2:w*3] = rgb_mask_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Original Approach", (w + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "RGB Approach", (w*2 + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Save the side-by-side comparison
        side_by_side_path = os.path.join(comparison_dir, f"{image_filename}_side_by_side.jpg")
        cv2.imwrite(side_by_side_path, side_by_side)
        print(f"Saved side-by-side comparison: {side_by_side_path}")
        
        return comparison_path, side_by_side_path
    else:
        print("Could not create comparison - one or both approaches failed.")
        return None, None

def main():
    # Define input and output directories
    input_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/jpeg"
    output_dir = "data/approach_comparison"
    
    # Create output directory
    create_directory(output_dir)
    
    # Process specific test images
    test_images = [
        os.path.join(input_dir, "page_20.jpeg"),
        os.path.join(input_dir, "page_13.jpeg")
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            compare_approaches(image_path, output_dir)
        else:
            print(f"Warning: Image not found: {image_path}")

if __name__ == "__main__":
    main() 