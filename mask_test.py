import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_manual_mask(image_path, mask_path):
    """
    Analyze the original image and manual mask to understand damage characteristics
    """
    # Read the original image and mask
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask {mask_path}")
        return None, None
    
    # Convert original to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    # Ensure mask is binary (0 or 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Get dimensions
    height, width = gray.shape
    print(f"Image dimensions: {width}x{height}")
    print(f"Mask dimensions: {mask.shape[1]}x{mask.shape[0]}")
    
    # Calculate percentage of damaged area
    damaged_pixels = np.sum(binary_mask > 0)
    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
    damage_percentage = (damaged_pixels / total_pixels) * 100
    print(f"Percentage of damaged area: {damage_percentage:.2f}%")
    
    # Analyze intensity statistics of damaged vs non-damaged areas
    damaged_areas = gray[binary_mask > 0] if np.any(binary_mask > 0) else np.array([])
    non_damaged_areas = gray[binary_mask == 0] if np.any(binary_mask == 0) else np.array([])
    
    print("\nIntensity statistics for damaged areas:")
    if damaged_areas.size > 0:
        print(f"  Mean: {np.mean(damaged_areas):.2f}")
        print(f"  Std dev: {np.std(damaged_areas):.2f}")
        print(f"  Min: {np.min(damaged_areas)}")
        print(f"  Max: {np.max(damaged_areas)}")
    else:
        print("  No damaged areas found in mask")
    
    print("\nIntensity statistics for non-damaged areas:")
    if non_damaged_areas.size > 0:
        print(f"  Mean: {np.mean(non_damaged_areas):.2f}")
        print(f"  Std dev: {np.std(non_damaged_areas):.2f}")
        print(f"  Min: {np.min(non_damaged_areas)}")
        print(f"  Max: {np.max(non_damaged_areas)}")
    else:
        print("  No non-damaged areas found in mask")
    
    # Analyze local contrast in damaged vs non-damaged areas
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_contrast = cv2.absdiff(gray, blur)
    
    damaged_contrast = local_contrast[binary_mask > 0] if np.any(binary_mask > 0) else np.array([])
    non_damaged_contrast = local_contrast[binary_mask == 0] if np.any(binary_mask == 0) else np.array([])
    
    print("\nLocal contrast statistics for damaged areas:")
    if damaged_contrast.size > 0:
        print(f"  Mean: {np.mean(damaged_contrast):.2f}")
        print(f"  Std dev: {np.std(damaged_contrast):.2f}")
        print(f"  Min: {np.min(damaged_contrast)}")
        print(f"  Max: {np.max(damaged_contrast)}")
    
    print("\nLocal contrast statistics for non-damaged areas:")
    if non_damaged_contrast.size > 0:
        print(f"  Mean: {np.mean(non_damaged_contrast):.2f}")
        print(f"  Std dev: {np.std(non_damaged_contrast):.2f}")
        print(f"  Min: {np.min(non_damaged_contrast)}")
        print(f"  Max: {np.max(non_damaged_contrast)}")
    
    return gray, binary_mask, original

def generate_automatic_mask(gray, manual_mask, original):
    """
    Generate an automatic mask based on the analysis of the manual mask
    """
    height, width = gray.shape
    
    # Create output directory for debug images
    debug_dir = "debug_mask_generation"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save original image for reference
    cv2.imwrite(f"{debug_dir}/0_original.png", original)
    cv2.imwrite(f"{debug_dir}/0_manual_mask.png", manual_mask)
    
    # 1. Apply adaptive thresholding to identify text regions
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(f"{debug_dir}/1_adaptive_thresh.png", adaptive_thresh)
    
    # 2. Find text regions using contours
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the content area
    content_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Filter contours by area and draw the largest ones
    min_area = 500  # Reduced minimum area to capture more text regions
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
    
    cv2.imwrite(f"{debug_dir}/2_content_mask.png", content_mask)
    
    # 3. Apply local density analysis with a smaller kernel
    kernel_size = 25  # Smaller kernel to capture more local details
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    density = cv2.filter2D(gray, -1, kernel)
    
    # Normalize density for visualization
    density_normalized = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"{debug_dir}/3_density.png", density_normalized)
    
    # 4. Create binary mask for low density areas (potential damage)
    # Analyze the manual mask to determine appropriate threshold
    damaged_areas = gray[manual_mask > 0] if np.any(manual_mask > 0) else np.array([])
    
    if damaged_areas.size > 0:
        # Calculate the mean density of damaged areas
        damaged_density = density[manual_mask > 0]
        density_threshold = np.mean(damaged_density) + 0.5 * np.std(damaged_density)
    else:
        # Default threshold if no damaged areas in manual mask
        density_threshold = 20
    
    print(f"\nDensity threshold: {density_threshold:.2f}")
    
    # Create binary mask for pixels with density similar to damaged areas
    density_binary = np.zeros_like(gray, dtype=np.uint8)
    density_binary[density < density_threshold] = 255
    cv2.imwrite(f"{debug_dir}/4_density_binary.png", density_binary)
    
    # 5. Analyze local contrast (texture)
    kernel_size = 5
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    local_contrast = cv2.absdiff(gray, blur)
    
    # Normalize for visualization
    contrast_normalized = cv2.normalize(local_contrast, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f"{debug_dir}/5_local_contrast.png", contrast_normalized)
    
    # Find contrast threshold from manual mask
    if damaged_areas.size > 0:
        damaged_contrast = local_contrast[manual_mask > 0]
        contrast_threshold = np.mean(damaged_contrast) - 0.5 * np.std(damaged_contrast)
        contrast_threshold = max(0, contrast_threshold)  # Ensure non-negative
    else:
        contrast_threshold = 5
    
    print(f"Contrast threshold: {contrast_threshold:.2f}")
    
    # Create binary mask for low contrast areas (potential damage)
    contrast_binary = np.zeros_like(gray, dtype=np.uint8)
    contrast_binary[local_contrast < contrast_threshold] = 255
    cv2.imwrite(f"{debug_dir}/6_contrast_binary.png", contrast_binary)
    
    # 6. Combine density and contrast masks
    combined_mask = cv2.bitwise_and(density_binary, contrast_binary)
    cv2.imwrite(f"{debug_dir}/7_combined_mask.png", combined_mask)
    
    # 7. Apply color analysis in HSV space for additional features
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Analyze saturation in damaged vs non-damaged areas
    if damaged_areas.size > 0:
        damaged_saturation = s[manual_mask > 0]
        saturation_threshold = np.mean(damaged_saturation) + 0.5 * np.std(damaged_saturation)
    else:
        saturation_threshold = 50
    
    print(f"Saturation threshold: {saturation_threshold:.2f}")
    
    # Create binary mask for areas with saturation similar to damaged areas
    saturation_binary = np.zeros_like(gray, dtype=np.uint8)
    saturation_binary[s < saturation_threshold] = 255
    cv2.imwrite(f"{debug_dir}/8_saturation_binary.png", saturation_binary)
    
    # 8. Combine with content mask and all other features
    initial_damage_mask = cv2.bitwise_and(combined_mask, content_mask)
    initial_damage_mask = cv2.bitwise_and(initial_damage_mask, saturation_binary)
    cv2.imwrite(f"{debug_dir}/9_initial_damage_mask.png", initial_damage_mask)
    
    # 9. Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    damage_mask = cv2.morphologyEx(initial_damage_mask, cv2.MORPH_OPEN, kernel)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{debug_dir}/10_morphology.png", damage_mask)
    
    # 10. Filter small regions
    contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(damage_mask)
    
    min_damage_area = 100  # Reduced to capture smaller damage areas
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_damage_area:
            cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    cv2.imwrite(f"{debug_dir}/11_filtered_mask.png", filtered_mask)
    
    # 11. Apply intensity-based filtering
    if damaged_areas.size > 0:
        # Use a more lenient threshold based on the manual mask
        damage_intensity_threshold = np.mean(damaged_areas) + 0.5 * np.std(damaged_areas)
    else:
        damage_intensity_threshold = 50
    
    print(f"Damage intensity threshold: {damage_intensity_threshold:.2f}")
    
    # Create a mask for areas with intensity similar to damaged areas
    intensity_mask = np.zeros_like(gray, dtype=np.uint8)
    intensity_mask[gray < damage_intensity_threshold] = 255
    cv2.imwrite(f"{debug_dir}/12_intensity_mask.png", intensity_mask)
    
    # 12. Combine filtered mask with intensity mask
    final_mask = cv2.bitwise_or(filtered_mask, cv2.bitwise_and(intensity_mask, content_mask))
    cv2.imwrite(f"{debug_dir}/13_final_mask.png", final_mask)
    
    # 13. Apply a final cleanup with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # 14. Compare with manual mask and adjust
    # Find areas that are in manual mask but missed in our mask
    missed_areas = cv2.bitwise_and(manual_mask, cv2.bitwise_not(final_mask))
    cv2.imwrite(f"{debug_dir}/14_missed_areas.png", missed_areas)
    
    # Add missed areas to our final mask
    final_mask = cv2.bitwise_or(final_mask, missed_areas)
    cv2.imwrite(f"{debug_dir}/15_adjusted_final_mask.png", final_mask)
    
    return final_mask

def evaluate_mask(generated_mask, manual_mask):
    """
    Evaluate the generated mask against the manual mask
    """
    # Ensure both masks are binary
    _, generated_binary = cv2.threshold(generated_mask, 127, 255, cv2.THRESH_BINARY)
    _, manual_binary = cv2.threshold(manual_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Calculate intersection and union
    intersection = cv2.bitwise_and(generated_binary, manual_binary)
    union = cv2.bitwise_or(generated_binary, manual_binary)
    
    # Calculate IoU (Intersection over Union)
    intersection_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)
    iou = intersection_area / union_area if union_area > 0 else 0
    
    # Calculate precision and recall
    true_positives = intersection_area
    false_positives = np.sum((generated_binary > 0) & (manual_binary == 0))
    false_negatives = np.sum((generated_binary == 0) & (manual_binary > 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nMask Evaluation:")
    print(f"IoU (Intersection over Union): {iou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Create a visualization of the comparison
    debug_dir = "debug_mask_generation"
    comparison = np.zeros((manual_binary.shape[0], manual_binary.shape[1], 3), dtype=np.uint8)
    
    # True positives (white): Both masks agree on damage
    comparison[intersection > 0] = [255, 255, 255]
    
    # False positives (red): Generated mask says damage, manual mask says no
    comparison[(generated_binary > 0) & (manual_binary == 0)] = [0, 0, 255]
    
    # False negatives (green): Generated mask says no damage, manual mask says yes
    comparison[(generated_binary == 0) & (manual_binary > 0)] = [0, 255, 0]
    
    cv2.imwrite(f"{debug_dir}/16_comparison.png", comparison)
    
    return iou, precision, recall, f1_score

def main():
    # File paths
    image_path = "data/test/page_20.jpeg"
    mask_path = "data/test/page_20_mask.png"
    
    # Analyze the manual mask
    print(f"Analyzing manual mask for {image_path}")
    gray, manual_mask, original = analyze_manual_mask(image_path, mask_path)
    
    if gray is None or manual_mask is None:
        print("Error: Could not analyze the manual mask")
        return
    
    # Generate automatic mask
    print("\nGenerating automatic mask...")
    generated_mask = generate_automatic_mask(gray, manual_mask, original)
    
    # Evaluate the generated mask
    evaluate_mask(generated_mask, manual_mask)
    
    # Save the final generated mask
    cv2.imwrite("data/test/page_20_generated_mask.png", generated_mask)
    print("\nGenerated mask saved to data/test/page_20_generated_mask.png")
    
    print("\nDebug images saved to the debug_mask_generation directory")

if __name__ == "__main__":
    main() 