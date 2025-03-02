import cv2
import numpy as np
import os
from pathlib import Path

def generate_damage_mask(image_path, output_path, overlay_path, debug_dir, min_hole_area=5000):
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {image_path}")
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"Image dimensions: {w}x{h}")
    cv2.imwrite(str(debug_dir / "1_gray.jpg"), gray)
    
    # Get the background color by sampling the corners
    corner_regions = [
        gray[0:50, 0:50],
        gray[0:50, w-50:w],
        gray[h-50:h, 0:50],
        gray[h-50:h, w-50:w]
    ]
    background_color = int(np.median([np.median(region) for region in corner_regions]))
    
    # Create binary image to separate page from background
    _, page_binary = cv2.threshold(gray, background_color - 30, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(str(debug_dir / "2_page_binary.jpg"), page_binary)
    
    # Find the page contour
    contours, _ = cv2.findContours(page_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_contour = max(contours, key=cv2.contourArea)
    
    # Create page mask
    page_mask = np.zeros_like(gray)
    cv2.drawContours(page_mask, [page_contour], -1, (255), -1)
    cv2.imwrite(str(debug_dir / "3_page_mask.jpg"), page_mask)
    
    # Apply CLAHE to enhance text contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(str(debug_dir / "3a_clahe.jpg"), enhanced)
    
    # Get text mask using adaptive threshold
    text_mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,  # Smaller block size for text
        10
    )
    text_mask = cv2.bitwise_and(text_mask, page_mask)
    cv2.imwrite(str(debug_dir / "3b_text_mask.jpg"), text_mask)
    
    # Find text regions
    text_kernel = np.ones((25, 25), np.uint8)
    text_regions = cv2.dilate(text_mask, text_kernel)
    cv2.imwrite(str(debug_dir / "3c_text_regions.jpg"), text_regions)
    
    # Create content area mask by finding the main text block
    content_area_mask = np.zeros_like(gray)
    text_contours, _ = cv2.findContours(text_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("\nText Region Analysis:")
    print(f"Number of text regions detected: {len(text_contours)}")
    
    if len(text_contours) > 0:
        # Sort contours by area
        text_contours_with_area = [(cv2.contourArea(c), c) for c in text_contours]
        text_contours_with_area.sort(reverse=True)
        
        print("\nTop 3 text regions by area:")
        for i, (area, contour) in enumerate(text_contours_with_area[:3]):
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Region {i+1}:")
            print(f"  Area: {area:.0f} pixels")
            print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
            print(f"  Aspect ratio: {w/h:.2f}")
            print(f"  % of page width: {(w/gray.shape[1]*100):.1f}%")
            print(f"  % of page height: {(h/gray.shape[0]*100):.1f}%")
        
        main_text_contour = text_contours_with_area[0][1]
        x, y, w, h = cv2.boundingRect(main_text_contour)
        
        # Calculate margin sizes relative to page dimensions
        margin = 50
        margin_percent_w = (margin / gray.shape[1]) * 100
        margin_percent_h = (margin / gray.shape[0]) * 100
        print(f"\nMargin analysis:")
        print(f"Current margin: {margin} pixels")
        print(f"Margin as % of width: {margin_percent_w:.1f}%")
        print(f"Margin as % of height: {margin_percent_h:.1f}%")
        
        # Original content box
        print(f"\nContent box before margin:")
        print(f"x: {x} to {x+w} ({(w/gray.shape[1]*100):.1f}% of width)")
        print(f"y: {y} to {y+h} ({(h/gray.shape[0]*100):.1f}% of height)")
        
        # Expand the content area
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2*margin)
        h = min(gray.shape[0] - y, h + 2*margin)
        
        print(f"\nContent box after margin:")
        print(f"x: {x} to {x+w} ({(w/gray.shape[1]*100):.1f}% of width)")
        print(f"y: {y} to {y+h} ({(h/gray.shape[0]*100):.1f}% of height)")
        
        # Draw the content area
        cv2.rectangle(content_area_mask, (x, y), (x+w, y+h), 255, -1)
        
        # Save visualization of content box
        content_box_vis = img.copy()
        cv2.rectangle(content_box_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / "3d_content_box_visualization.jpg"), content_box_vis)
    
    content_area_mask = cv2.bitwise_and(content_area_mask, page_mask)
    cv2.imwrite(str(debug_dir / "3d_content_area_mask.jpg"), content_area_mask)
    
    # Apply local adaptive threshold with more aggressive parameters
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        45,  # Larger block size
        12   # More aggressive C value
    )
    
    # Apply both page mask and content area mask
    content_mask = cv2.bitwise_and(adaptive_thresh, content_area_mask)
    cv2.imwrite(str(debug_dir / "4_content_mask.jpg"), content_mask)
    
    # Calculate local density with larger kernel
    kernel_size = 99  # Changed to odd number
    density = cv2.GaussianBlur(content_mask.astype(float), (kernel_size, kernel_size), 0)
    
    # Analyze density values
    density_min = float(np.min(density))  # Convert to scalar
    density_max = float(np.max(density))  # Convert to scalar
    density_mean = float(np.mean(density))  # Convert to scalar
    density_std = float(np.std(density))  # Convert to scalar
    
    print("\nDensity Analysis:")
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    print(f"Density stats:")
    print(f"  Min: {density_min:.2f}")
    print(f"  Max: {density_max:.2f}")
    print(f"  Mean: {density_mean:.2f}")
    print(f"  Std dev: {density_std:.2f}")
    
    # Sample density in different regions
    margin_width = 100
    center_width = 100
    # Left margin
    left_margin = density[margin_width:-margin_width, :margin_width]
    # Right margin
    right_margin = density[margin_width:-margin_width, -margin_width:]
    # Center of page
    center_x = density.shape[1] // 2
    center_region = density[margin_width:-margin_width, 
                          center_x-center_width//2:center_x+center_width//2]
    
    print("\nRegion density stats:")
    print("Left margin:")
    print(f"  Mean: {left_margin.mean():.2f}")
    print(f"  Std dev: {left_margin.std():.2f}")
    print("Right margin:")
    print(f"  Mean: {right_margin.mean():.2f}")
    print(f"  Std dev: {right_margin.std():.2f}")
    print("Center region:")
    print(f"  Mean: {center_region.mean():.2f}")
    print(f"  Std dev: {center_region.std():.2f}")
    
    # Normalize density
    density = (density / density.max() * 255).astype(np.uint8)
    cv2.imwrite(str(debug_dir / "5_density.jpg"), density)
    
    # Create edge mask
    edge_kernel = np.ones((25, 25), np.uint8)
    edge_mask = cv2.morphologyEx(page_mask, cv2.MORPH_GRADIENT, edge_kernel)
    cv2.imwrite(str(debug_dir / "6_edge_mask.jpg"), edge_mask)
    
    # Threshold density more aggressively
    current_threshold = 20
    print(f"\nDensity threshold: {current_threshold}")
    print(f"Pixels below threshold: {(density < current_threshold).sum()}")
    print(f"Percentage of page below threshold: {(density < current_threshold).sum() / density.size * 100:.1f}%")
    
    _, damage_binary = cv2.threshold(density, current_threshold, 255, cv2.THRESH_BINARY_INV)
    damage_binary = cv2.bitwise_and(damage_binary, content_area_mask)
    cv2.imwrite(str(debug_dir / "7_damage_binary.jpg"), damage_binary)
    
    # Clean up noise more aggressively
    kernel_small = np.ones((7,7), np.uint8)
    kernel_medium = np.ones((21,21), np.uint8)
    damage_binary = cv2.morphologyEx(damage_binary, cv2.MORPH_OPEN, kernel_small)
    damage_binary = cv2.morphologyEx(damage_binary, cv2.MORPH_CLOSE, kernel_medium)
    cv2.imwrite(str(debug_dir / "8_cleaned_binary.jpg"), damage_binary)
    
    # Find damage contours
    damage_contours, _ = cv2.findContours(damage_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask and overlay
    final_mask = np.zeros_like(gray)
    overlay = img.copy()
    
    for contour in damage_contours:
        area = cv2.contourArea(contour)
        if area > min_hole_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # More sophisticated filtering:
            # 1. Area check (already done)
            # 2. Shape checks
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.1 < aspect_ratio < 10:
                # 3. Check if the region has significant intensity variation
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, (255), -1)
                roi = cv2.bitwise_and(gray, mask)
                if roi.any():  # Check if ROI is not empty
                    roi_valid = roi[roi != 0]  # Get non-zero pixels
                    std_dev = np.std(roi_valid)
                    mean_intensity = np.mean(roi_valid)
                    
                    print(f"\nAnalyzing potential damage region:")
                    print(f"Position: x={x}, y={y}, w={w}, h={h}")
                    print(f"Area: {area} pixels")
                    print(f"Aspect ratio: {aspect_ratio:.2f}")
                    print(f"Intensity stats:")
                    print(f"  Mean: {mean_intensity:.1f}")
                    print(f"  Std dev: {std_dev:.1f}")
                    print(f"  Min: {roi_valid.min()}")
                    print(f"  Max: {roi_valid.max()}")
                    print(f"  Is damage: {std_dev > 25 or mean_intensity < 100}")
                    
                    # Only include regions with either:
                    # - High intensity variation (damaged areas)
                    # - Very dark regions (holes/excised areas)
                    if std_dev > 25 or mean_intensity < 100:
                        cv2.drawContours(final_mask, [contour], -1, (255), -1)
                        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
                        
                        # Save region visualization
                        region_vis = img.copy()
                        cv2.drawContours(region_vis, [contour], -1, (0, 255, 0), 2)
                        region_name = f"region_x{x}_y{y}_damage.jpg"
                        cv2.imwrite(str(debug_dir / region_name), region_vis)
                    else:
                        # Save rejected region for analysis
                        region_vis = img.copy()
                        cv2.drawContours(region_vis, [contour], -1, (255, 0, 0), 2)
                        region_name = f"region_x{x}_y{y}_rejected.jpg"
                        cv2.imwrite(str(debug_dir / region_name), region_vis)
    
    # Optional: One final cleanup pass to smooth edges
    kernel_medium = np.ones((21,21), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium)
    cv2.imwrite(str(debug_dir / "9_final_mask.jpg"), final_mask)
    
    # Create final overlay
    alpha = 0.5
    overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    cv2.imwrite(str(output_path), final_mask)
    cv2.imwrite(str(overlay_path), overlay_img)
    
    return final_mask, overlay_img

def process_folder():
    # Set up paths
    base_path = Path("data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate")
    input_dir = base_path / "jpeg"
    mask_dir = base_path / "masks"
    overlay_dir = base_path / "overlays"
    debug_dir = base_path / "debug"
    
    # Create output directories
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all jpg files
    for img_path in input_dir.glob("*.jp*g"):  # Handle both .jpg and .jpeg
        print(f"\nProcessing {img_path}")
        
        # Create debug directory for this image
        img_debug_dir = debug_dir / img_path.stem
        mask_path = mask_dir / f"{img_path.stem}_mask.jpg"
        overlay_path = overlay_dir / f"{img_path.stem}_overlay.jpg"
        
        try:
            mask, overlay = generate_damage_mask(
                img_path, 
                mask_path, 
                overlay_path,
                img_debug_dir
            )
            if mask is None:
                print(f"Failed to process {img_path}")
            else:
                print(f"Successfully processed {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def is_damage_region(mean_intensity, std_dev, area):
    # Updated damage detection criteria based on analysis
    is_damage = (
        (mean_intensity < 20 and std_dev < 7) or  # Very dark, uniform regions
        (mean_intensity < 150 and std_dev > 15) or  # Dark with high variation
        (mean_intensity < 130 and area < 50000) or  # Small dark regions
        (mean_intensity < 140 and std_dev > 10 and area < 100000)  # Medium dark regions with variation
    )
    
    return is_damage

def analyze_potential_damage(img, x, y, w, h, area):
    region = img[y:y+h, x:x+w]
    mean_intensity = float(np.mean(region))
    std_dev = float(np.std(region))
    min_val = float(np.min(region))
    max_val = float(np.max(region))
    
    is_damage = is_damage_region(mean_intensity, std_dev, area)
    
    print(f"\nAnalyzing potential damage region:")
    print(f"Position: x={x}, y={y}, w={w}, h={h}")
    print(f"Area: {area} pixels")
    print(f"Aspect ratio: {w/h:.2f}")
    print(f"Intensity stats:")
    print(f"  Mean: {mean_intensity:.1f}")
    print(f"  Std dev: {std_dev:.1f}")
    print(f"  Min: {min_val}")
    print(f"  Max: {max_val}")
    print(f"  Is damage: {is_damage}")
    
    return is_damage

def find_text_regions(gray):
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and get bounding boxes
    min_area = 5000  # Increased minimum area to filter out noise
    text_regions = []
    
    # Calculate total image area for percentage-based filtering
    image_area = gray.shape[0] * gray.shape[1]
    min_percentage = 0.01  # Minimum 1% of page area to be considered text
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Only include regions that are at least 1% of the image area
        if area > min_area and area > (image_area * min_percentage):
            x, y, w, h = cv2.boundingRect(contour)
            text_regions.append((x, y, w, h))
    
    # Sort text regions by area in descending order
    text_regions.sort(key=lambda x: x[2] * x[3], reverse=True)
    
    return text_regions

def process_image(image_path):
    print(f"\nProcessing {image_path}")
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Print image dimensions
    height, width = gray.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Find text regions
    text_regions = find_text_regions(gray)
    
    print("\nText Region Analysis:")
    print(f"Number of text regions detected: {len(text_regions)}")
    
    # Check if this is a blank page (no significant text regions)
    if len(text_regions) == 0:
        print(f"This appears to be a blank page. Skipping further processing.")
        print(f"Successfully processed {image_path}")
        return
    
    # For pages with many text regions (likely blank or problematic pages), skip processing
    if len(text_regions) > 5:
        print(f"This page has {len(text_regions)} text regions, which suggests it may be blank or have noise.")
        print(f"Skipping detailed processing for {image_path}")
        print(f"Successfully processed {image_path}")
        return
    
    # Sort text regions by area (largest first)
    text_regions.sort(key=lambda x: x[2] * x[3], reverse=True)
    
    # Print details for the top 3 text regions (or fewer if less than 3 exist)
    print("\nTop 3 text regions by area:")
    for i in range(min(3, len(text_regions))):
        x, y, w, h = text_regions[i]
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        print(f"Region {i+1}:")
        print(f"  Area: {area} pixels")
        print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
        print(f"  Aspect ratio: {aspect_ratio:.2f}")
        print(f"  % of page width: {w/width*100:.1f}%")
        print(f"  % of page height: {h/height*100:.1f}%")
    
    # Use the largest text region as the content box
    content_x, content_y, content_w, content_h = text_regions[0]
    
    # Add margin to content box
    margin = 50  # pixels
    print("\nMargin analysis:")
    print(f"Current margin: {margin} pixels")
    print(f"Margin as % of width: {margin/width*100:.1f}%")
    print(f"Margin as % of height: {margin/height*100:.1f}%")
    
    print("\nContent box before margin:")
    print(f"x: {content_x} to {content_x + content_w} ({content_w/width*100:.1f}% of width)")
    print(f"y: {content_y} to {content_y + content_h} ({content_h/height*100:.1f}% of height)")
    
    # Adjust content box with margin, ensuring we don't go outside image boundaries
    content_x = max(0, content_x - margin)
    content_y = max(0, content_y - margin)
    content_w = min(width - content_x, content_w + margin)
    content_h = min(height - content_y, content_h + margin)
    
    print("\nContent box after margin:")
    print(f"x: {content_x} to {content_x + content_w} ({content_w/width*100:.1f}% of width)")
    print(f"y: {content_y} to {content_y + content_h} ({content_h/height*100:.1f}% of height)")
    
    # Create a mask for the content area
    content_mask = np.zeros((height, width), dtype=np.uint8)
    content_mask[content_y:content_y+content_h, content_x:content_x+content_w] = 255
    
    # Analyze density
    print("\nDensity Analysis:")
    kernel_size = 99
    print(f"Kernel size: {kernel_size}x{kernel_size}")
    
    # Calculate local density using a large kernel
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    density = cv2.filter2D(gray, -1, kernel)
    
    # Calculate density statistics
    density_min = float(np.min(density))
    density_max = float(np.max(density))
    density_mean = float(np.mean(density))
    density_std = float(np.std(density))
    
    print("Density stats:")
    print(f"  Min: {density_min:.2f}")
    print(f"  Max: {density_max:.2f}")
    print(f"  Mean: {density_mean:.2f}")
    print(f"  Std dev: {density_std:.2f}")
    
    # Calculate density statistics for different regions
    # Left margin
    left_margin = gray[:, :content_x] if content_x > 0 else np.zeros((1, 1), dtype=np.uint8)
    left_mean = float(np.mean(left_margin)) if left_margin.size > 0 else 0
    left_std = float(np.std(left_margin)) if left_margin.size > 0 else 0
    
    # Right margin
    right_margin = gray[:, content_x+content_w:] if content_x+content_w < width else np.zeros((1, 1), dtype=np.uint8)
    right_mean = float(np.mean(right_margin)) if right_margin.size > 0 else 0
    right_std = float(np.std(right_margin)) if right_margin.size > 0 else 0
    
    # Center region
    center = gray[content_y:content_y+content_h, content_x:content_x+content_w]
    center_mean = float(np.mean(center))
    center_std = float(np.std(center))
    
    print("\nRegion density stats:")
    print("Left margin:")
    print(f"  Mean: {left_mean:.2f}")
    print(f"  Std dev: {left_std:.2f}")
    print("Right margin:")
    print(f"  Mean: {right_mean:.2f}")
    print(f"  Std dev: {right_std:.2f}")
    print("Center region:")
    print(f"  Mean: {center_mean:.2f}")
    print(f"  Std dev: {center_std:.2f}")
    
    # Threshold for density
    density_threshold = 20
    print(f"\nDensity threshold: {density_threshold}")
    
    # Create binary mask for pixels below threshold
    density_binary = np.zeros_like(density, dtype=np.uint8)
    density_binary[density < density_threshold] = 255
    
    # Count pixels below threshold within content area
    content_mask_bool = content_mask > 0
    density_binary_bool = density_binary > 0
    
    # Use logical AND to find pixels that are both in content area and below threshold
    pixels_below_threshold = np.logical_and(content_mask_bool, density_binary_bool)
    count_below_threshold = int(np.sum(pixels_below_threshold))
    
    # Calculate percentage
    total_pixels = width * height
    percentage_below = count_below_threshold / total_pixels * 100
    
    print(f"Pixels below threshold: {count_below_threshold}")
    print(f"Percentage of page below threshold: {percentage_below:.1f}%")
    
    # Create damage binary mask
    damage_binary = np.zeros_like(gray, dtype=np.uint8)
    damage_binary[pixels_below_threshold] = 255
    
    # Find contours in the damage binary
    contours, _ = cv2.findContours(damage_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_damage_area = 5000
    damage_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_damage_area:
            x, y, w, h = cv2.boundingRect(contour)
            damage_regions.append((x, y, w, h, area))
    
    # Analyze each potential damage region
    for x, y, w, h, area in damage_regions:
        print("\nAnalyzing potential damage region:")
        print(f"Position: x={x}, y={y}, w={w}, h={h}")
        print(f"Area: {area:.1f} pixels")
        
        aspect_ratio = w / h if h > 0 else 0
        print(f"Aspect ratio: {aspect_ratio:.2f}")
        
        # Extract the region from the grayscale image
        region = gray[y:y+h, x:x+w]
        
        # Calculate intensity statistics
        mean_intensity = float(np.mean(region))
        std_intensity = float(np.std(region))
        min_intensity = float(np.min(region))
        max_intensity = float(np.max(region))
        
        print("Intensity stats:")
        print(f"  Mean: {mean_intensity:.1f}")
        print(f"  Std dev: {std_intensity:.1f}")
        print(f"  Min: {int(min_intensity)}")
        print(f"  Max: {int(max_intensity)}")
        
        # Determine if this is damage
        is_damage = mean_intensity < 50
        print(f"  Is damage: {is_damage}")
    
    print(f"Successfully processed {image_path}")
    return True

if __name__ == "__main__":
    process_folder()