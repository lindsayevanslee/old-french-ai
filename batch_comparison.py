#!/usr/bin/env python3
# batch_comparison.py
# A script to compare the performance of all three approaches on a batch of images

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse
import shutil
from tqdm import tqdm

# Import functions from all three approaches
from generate_damage_masks import generate_damage_mask
from rgb_damage_detection import detect_damage_rgb
from combined_damage_detection import combined_damage_detection, create_directory

def compare_approaches(image_paths, output_dir="data/batch_comparison", num_images=None, debug=False):
    """
    Compare all three approaches on a batch of images
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save comparison results
        num_images: Number of images to process (None for all)
        debug: Whether to save debug images
    """
    # Create output directories
    create_directory(output_dir)
    
    # Create a CSV file for metrics
    metrics_file = os.path.join(output_dir, "metrics.csv")
    with open(metrics_file, "w") as f:
        f.write("image,original_pixels,rgb_pixels,combined_pixels,original_time,rgb_time,combined_time\n")
    
    # Process a subset of images if specified
    if num_images is not None and num_images > 0:
        image_paths = image_paths[:num_images]
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        image_filename = Path(image_path).stem
        
        # Create image-specific output directory
        image_output_dir = os.path.join(output_dir, image_filename)
        create_directory(image_output_dir)
        
        # Create temporary directories for each approach
        temp_original_dir = os.path.join(image_output_dir, "original")
        temp_rgb_dir = os.path.join(image_output_dir, "rgb")
        temp_combined_dir = os.path.join(image_output_dir, "combined")
        
        create_directory(temp_original_dir)
        create_directory(temp_rgb_dir)
        create_directory(temp_combined_dir)
        
        # Create subdirectories for each approach
        for approach_dir in [temp_original_dir, temp_rgb_dir, temp_combined_dir]:
            create_directory(os.path.join(approach_dir, "masks"))
            create_directory(os.path.join(approach_dir, "overlays"))
            if debug:
                create_directory(os.path.join(approach_dir, "debug", image_filename))
        
        # Run the original approach
        original_start_time = time.time()
        
        # Set up debug directory only if debug is True
        original_debug_dir = None
        if debug:
            original_debug_dir = os.path.join(temp_original_dir, "debug", image_filename)
        
        original_mask_path, original_overlay_path = generate_damage_mask(
            image_path,
            os.path.join(temp_original_dir, "masks"),
            os.path.join(temp_original_dir, "overlays"),
            original_debug_dir
        )
        original_time = time.time() - original_start_time
        
        # Run the RGB-based approach
        rgb_start_time = time.time()
        rgb_mask_path, rgb_overlay_path = detect_damage_rgb(
            image_path,
            temp_rgb_dir
        )
        rgb_time = time.time() - rgb_start_time
        
        # Run the combined approach
        combined_start_time = time.time()
        combined_mask_path, combined_overlay_path = combined_damage_detection(
            image_path,
            temp_combined_dir,
            debug
        )
        combined_time = time.time() - combined_start_time
        
        # Read the masks
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        rgb_mask = cv2.imread(rgb_mask_path, cv2.IMREAD_GRAYSCALE)
        combined_mask = cv2.imread(combined_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Count the number of pixels in each mask
        original_pixels = np.sum(original_mask > 0)
        rgb_pixels = np.sum(rgb_mask > 0)
        combined_pixels = np.sum(combined_mask > 0)
        
        # Write metrics to CSV
        with open(metrics_file, "a") as f:
            f.write(f"{image_filename},{original_pixels},{rgb_pixels},{combined_pixels},{original_time:.4f},{rgb_time:.4f},{combined_time:.4f}\n")
        
        # Create a side-by-side comparison
        original_image = cv2.imread(image_path)
        
        # Resize for better visualization
        scale = 0.5
        h, w = original_image.shape[:2]
        h_resized, w_resized = int(h * scale), int(w * scale)
        
        original_image_resized = cv2.resize(original_image, (w_resized, h_resized))
        original_mask_resized = cv2.resize(original_mask, (w_resized, h_resized))
        rgb_mask_resized = cv2.resize(rgb_mask, (w_resized, h_resized))
        combined_mask_resized = cv2.resize(combined_mask, (w_resized, h_resized))
        
        # Create a side-by-side comparison
        side_by_side = np.zeros((h_resized, w_resized * 4, 3), dtype=np.uint8)
        
        # Original image
        side_by_side[:, 0:w_resized] = original_image_resized
        
        # Original mask (convert to 3-channel)
        original_mask_color = cv2.cvtColor(original_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized:w_resized*2] = original_mask_color
        
        # RGB mask (convert to 3-channel)
        rgb_mask_color = cv2.cvtColor(rgb_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized*2:w_resized*3] = rgb_mask_color
        
        # Combined mask (convert to 3-channel)
        combined_mask_color = cv2.cvtColor(combined_mask_resized, cv2.COLOR_GRAY2BGR)
        side_by_side[:, w_resized*3:w_resized*4] = combined_mask_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original Image", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Original Approach", (w_resized + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "RGB Approach", (w_resized*2 + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Combined Approach", (w_resized*3 + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Add timing information
        cv2.putText(side_by_side, f"Time: {original_time:.2f}s", (w_resized + 10, h_resized - 20), font, 0.7, (255, 255, 255), 1)
        cv2.putText(side_by_side, f"Time: {rgb_time:.2f}s", (w_resized*2 + 10, h_resized - 20), font, 0.7, (255, 255, 255), 1)
        cv2.putText(side_by_side, f"Time: {combined_time:.2f}s", (w_resized*3 + 10, h_resized - 20), font, 0.7, (255, 255, 255), 1)
        
        # Save the side-by-side comparison
        comparison_path = os.path.join(image_output_dir, f"{image_filename}_comparison.jpg")
        cv2.imwrite(comparison_path, side_by_side)
        
        # Copy the original masks and overlays to the image output directory
        shutil.copy(original_mask_path, os.path.join(image_output_dir, f"{image_filename}_original_mask.jpg"))
        shutil.copy(original_overlay_path, os.path.join(image_output_dir, f"{image_filename}_original_overlay.jpg"))
        shutil.copy(rgb_mask_path, os.path.join(image_output_dir, f"{image_filename}_rgb_mask.jpg"))
        shutil.copy(rgb_overlay_path, os.path.join(image_output_dir, f"{image_filename}_rgb_overlay.jpg"))
        shutil.copy(combined_mask_path, os.path.join(image_output_dir, f"{image_filename}_combined_mask.jpg"))
        shutil.copy(combined_overlay_path, os.path.join(image_output_dir, f"{image_filename}_combined_overlay.jpg"))
    
    # Generate summary plots
    generate_summary_plots(metrics_file, output_dir)

def generate_summary_plots(metrics_file, output_dir):
    """
    Generate summary plots from the metrics CSV file
    
    Args:
        metrics_file: Path to the metrics CSV file
        output_dir: Directory to save the plots
    """
    import pandas as pd
    
    # Read the metrics CSV file
    df = pd.read_csv(metrics_file)
    
    # Create a plot of the number of pixels detected by each approach
    plt.figure(figsize=(12, 6))
    
    # Plot the number of pixels
    x = range(len(df))
    plt.bar(x, df["original_pixels"], width=0.25, label="Original Approach", alpha=0.7)
    plt.bar([i + 0.25 for i in x], df["rgb_pixels"], width=0.25, label="RGB Approach", alpha=0.7)
    plt.bar([i + 0.5 for i in x], df["combined_pixels"], width=0.25, label="Combined Approach", alpha=0.7)
    
    plt.xlabel("Image")
    plt.ylabel("Number of Pixels")
    plt.title("Number of Pixels Detected by Each Approach")
    plt.xticks([i + 0.25 for i in x], df["image"], rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "pixels_comparison.png"))
    
    # Create a plot of the processing time for each approach
    plt.figure(figsize=(12, 6))
    
    # Plot the processing time
    plt.bar(x, df["original_time"], width=0.25, label="Original Approach", alpha=0.7)
    plt.bar([i + 0.25 for i in x], df["rgb_time"], width=0.25, label="RGB Approach", alpha=0.7)
    plt.bar([i + 0.5 for i in x], df["combined_time"], width=0.25, label="Combined Approach", alpha=0.7)
    
    plt.xlabel("Image")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time for Each Approach")
    plt.xticks([i + 0.25 for i in x], df["image"], rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, "time_comparison.png"))
    
    # Calculate average metrics
    avg_original_pixels = df["original_pixels"].mean()
    avg_rgb_pixels = df["rgb_pixels"].mean()
    avg_combined_pixels = df["combined_pixels"].mean()
    
    avg_original_time = df["original_time"].mean()
    avg_rgb_time = df["rgb_time"].mean()
    avg_combined_time = df["combined_time"].mean()
    
    # Create a summary text file
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("Summary of Approach Comparison\n")
        f.write("=============================\n\n")
        f.write(f"Number of images processed: {len(df)}\n\n")
        
        f.write("Average number of pixels detected:\n")
        f.write(f"  Original Approach: {avg_original_pixels:.2f}\n")
        f.write(f"  RGB Approach: {avg_rgb_pixels:.2f}\n")
        f.write(f"  Combined Approach: {avg_combined_pixels:.2f}\n\n")
        
        f.write("Average processing time (seconds):\n")
        f.write(f"  Original Approach: {avg_original_time:.4f}\n")
        f.write(f"  RGB Approach: {avg_rgb_time:.4f}\n")
        f.write(f"  Combined Approach: {avg_combined_time:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Compare damage detection approaches on a batch of images")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="data/batch_comparison", 
                        help="Directory to save comparison results")
    parser.add_argument("--num_images", type=int, default=5, 
                        help="Number of images to process (default: 5, use 0 for all)")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    
    args = parser.parse_args()
    
    # Use default input directory if not specified
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/jpeg"
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} not found")
        return
    
    # Get all JPEG files in the input directory
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpeg', '.jpg')) and not f.startswith('._')]
    
    # Sort the image paths to ensure consistent ordering
    image_paths.sort()
    
    # Process the specified number of images
    num_images = args.num_images if args.num_images > 0 else None
    
    # Compare approaches
    compare_approaches(image_paths, args.output_dir, num_images, args.debug)

if __name__ == "__main__":
    main() 