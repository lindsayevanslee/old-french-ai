#!/usr/bin/env python3
# rgb_damage_detection.py
# A script to explore RGB-based approaches for manuscript damage detection

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

#output prints to data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/mask_methods/rbg

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_debug_image(image, filename, debug_dir):
    """Save an image to the debug directory"""
    filepath = os.path.join(debug_dir, filename)
    cv2.imwrite(filepath, image)
    print(f"Saved: {filepath}")

def plot_histograms(image, debug_dir):
    """Plot and save histograms for each RGB channel"""
    # Split the channels (OpenCV uses BGR order)
    b, g, r = cv2.split(image)
    
    plt.figure(figsize=(15, 5))
    
    # Plot histogram for each channel
    plt.subplot(131)
    plt.hist(r.ravel(), 256, [0, 256], color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(132)
    plt.hist(g.ravel(), 256, [0, 256], color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    
    plt.subplot(133)
    plt.hist(b.ravel(), 256, [0, 256], color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    
    plt.tight_layout()
    histogram_path = os.path.join(debug_dir, 'rgb_histograms.png')
    plt.savefig(histogram_path)
    plt.close()
    print(f"Saved: {histogram_path}")

def detect_damage_rgb(image_path, output_dir):
    """
    Detect damage in a manuscript image using RGB-based approaches
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output and debug images
    """
    # Extract filename without extension for output naming
    image_filename = Path(image_path).stem
    
    # Create output directories
    debug_dir = os.path.join(output_dir, 'debug', image_filename)
    masks_dir = os.path.join(output_dir, 'masks')
    overlays_dir = os.path.join(output_dir, 'overlays')
    
    create_directory(debug_dir)
    create_directory(masks_dir)
    create_directory(overlays_dir)
    
    # Read the input image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Save original image for reference
    save_debug_image(image, '01_original.jpg', debug_dir)
    
    # Generate histograms for analysis
    plot_histograms(image, debug_dir)
    
    # Split into RGB channels (OpenCV uses BGR order)
    blue, green, red = cv2.split(image)
    
    # Save individual channels for visualization
    save_debug_image(blue, '02_blue_channel.jpg', debug_dir)
    save_debug_image(green, '03_green_channel.jpg', debug_dir)
    save_debug_image(red, '04_red_channel.jpg', debug_dir)
    
    # APPROACH 1: CHANNEL THRESHOLDING
    # -----------------------------
    # Idea: Damaged areas often have different intensity distributions in each channel
    
    # Create binary masks for each channel using adaptive thresholding
    # Adaptive thresholding works well for documents with varying illumination
    blue_thresh = cv2.adaptiveThreshold(blue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    green_thresh = cv2.adaptiveThreshold(green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
    red_thresh = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    
    save_debug_image(blue_thresh, '05_blue_threshold.jpg', debug_dir)
    save_debug_image(green_thresh, '06_green_threshold.jpg', debug_dir)
    save_debug_image(red_thresh, '07_red_threshold.jpg', debug_dir)
    
    # APPROACH 2: CHANNEL RATIOS
    # -----------------------
    # Idea: Damage often creates imbalances between color channels
    
    # Calculate channel ratios (adding small epsilon to avoid division by zero)
    epsilon = 1e-10
    r_g_ratio = np.divide(red.astype(float), green.astype(float) + epsilon)
    r_b_ratio = np.divide(red.astype(float), blue.astype(float) + epsilon)
    g_b_ratio = np.divide(green.astype(float), blue.astype(float) + epsilon)
    
    # Normalize ratios to 0-255 range for visualization
    r_g_ratio_norm = cv2.normalize(r_g_ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r_b_ratio_norm = cv2.normalize(r_b_ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    g_b_ratio_norm = cv2.normalize(g_b_ratio, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    save_debug_image(r_g_ratio_norm, '08_red_green_ratio.jpg', debug_dir)
    save_debug_image(r_b_ratio_norm, '09_red_blue_ratio.jpg', debug_dir)
    save_debug_image(g_b_ratio_norm, '10_green_blue_ratio.jpg', debug_dir)
    
    # Threshold the ratios to identify abnormal areas
    # High red-to-blue ratio often indicates yellowing or browning damage
    _, r_b_ratio_mask = cv2.threshold(r_b_ratio_norm, 150, 255, cv2.THRESH_BINARY)
    save_debug_image(r_b_ratio_mask, '11_red_blue_ratio_mask.jpg', debug_dir)
    
    # APPROACH 3: CHANNEL DIFFERENCES
    # ----------------------------
    # Idea: Absolute differences between channels can highlight damage
    
    r_g_diff = cv2.absdiff(red, green)
    r_b_diff = cv2.absdiff(red, blue)
    g_b_diff = cv2.absdiff(green, blue)
    
    save_debug_image(r_g_diff, '12_red_green_diff.jpg', debug_dir)
    save_debug_image(r_b_diff, '13_red_blue_diff.jpg', debug_dir)
    save_debug_image(g_b_diff, '14_green_blue_diff.jpg', debug_dir)
    
    # Threshold the differences to create masks
    _, r_b_diff_mask = cv2.threshold(r_b_diff, 30, 255, cv2.THRESH_BINARY)
    save_debug_image(r_b_diff_mask, '15_red_blue_diff_mask.jpg', debug_dir)
    
    # APPROACH 4: COMBINED RGB ANALYSIS
    # ------------------------------
    # Idea: Combine multiple RGB-based indicators for better damage detection
    
    # Convert to grayscale for reference
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_debug_image(gray, '16_grayscale.jpg', debug_dir)
    
    # Create a mask for yellowing/browning (high red, medium green, low blue)
    # This is common in aged or water-damaged manuscripts
    yellowing_mask = np.zeros_like(gray)
    # Where red > green > blue with significant differences
    yellowing_mask[((red > green + 10) & (green > blue + 10))] = 255
    save_debug_image(yellowing_mask, '17_yellowing_mask.jpg', debug_dir)
    
    # Create a mask for potential ink damage
    # Ink typically has balanced RGB values (dark in all channels)
    # Damaged ink often becomes unbalanced
    potential_ink = np.zeros_like(gray)
    # Areas that are dark in all channels (potential ink)
    potential_ink[(red < 100) & (green < 100) & (blue < 100)] = 255
    save_debug_image(potential_ink, '18_potential_ink.jpg', debug_dir)
    
    # Detect ink damage by finding areas where ink is expected but channel balance is off
    ink_damage = np.zeros_like(gray)
    # Areas that are dark but have significant channel differences
    ink_damage[(potential_ink > 0) & ((r_b_diff > 30) | (r_g_diff > 30) | (g_b_diff > 30))] = 255
    save_debug_image(ink_damage, '19_ink_damage.jpg', debug_dir)
    
    # FINAL DAMAGE MASK CREATION
    # -----------------------
    # Combine different damage indicators into a final mask
    
    # Combine yellowing and ink damage masks
    combined_mask = cv2.bitwise_or(yellowing_mask, ink_damage)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    save_debug_image(combined_mask, '20_combined_mask.jpg', debug_dir)
    
    # Create a more aggressive mask by also including the r_b_ratio_mask
    aggressive_mask = cv2.bitwise_or(combined_mask, r_b_ratio_mask)
    aggressive_mask = cv2.morphologyEx(aggressive_mask, cv2.MORPH_OPEN, kernel)
    aggressive_mask = cv2.morphologyEx(aggressive_mask, cv2.MORPH_CLOSE, kernel)
    
    save_debug_image(aggressive_mask, '21_aggressive_mask.jpg', debug_dir)
    
    # Save final masks
    final_mask_path = os.path.join(masks_dir, f"{image_filename}_mask.jpg")
    cv2.imwrite(final_mask_path, aggressive_mask)
    print(f"Saved final mask: {final_mask_path}")
    
    # Create overlay visualization
    # Convert mask to 3 channel for overlay
    mask_3channel = cv2.cvtColor(aggressive_mask, cv2.COLOR_GRAY2BGR)
    # Create a red tint for damaged areas
    mask_3channel[:,:,0] = 0  # Blue channel
    mask_3channel[:,:,1] = 0  # Green channel
    # Red channel already has the mask values
    
    # Blend original image with the mask
    alpha = 0.3  # Transparency factor
    overlay = cv2.addWeighted(image, 1, mask_3channel, alpha, 0)
    
    # Save overlay
    overlay_path = os.path.join(overlays_dir, f"{image_filename}_overlay.jpg")
    cv2.imwrite(overlay_path, overlay)
    print(f"Saved overlay: {overlay_path}")
    
    return final_mask_path, overlay_path

def main():
    # Define input and output directories
    input_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/jpeg"
    output_dir = "data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate/mask_methods/rbg"
    
    # Create output directory
    create_directory(output_dir)
    
    # Process specific test images
    test_images = [
        os.path.join(input_dir, "page_20.jpeg"),
        os.path.join(input_dir, "page_13.jpeg")
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            detect_damage_rgb(image_path, output_dir)
        else:
            print(f"Warning: Image not found: {image_path}")

if __name__ == "__main__":
    main() 