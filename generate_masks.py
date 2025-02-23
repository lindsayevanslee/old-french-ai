import cv2
import numpy as np
import os
from pathlib import Path

def generate_damage_mask(image_path, output_path, overlay_path, debug_dir, min_hole_area=5000):
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(debug_dir / "1_gray.jpg"), gray)
    
    # Get the background color by sampling the corners
    h, w = gray.shape
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
    
    # Apply local adaptive threshold with more aggressive parameters
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        45,  # Larger block size
        12   # More aggressive C value
    )
    
    # Apply page mask and save content mask
    content_mask = cv2.bitwise_and(adaptive_thresh, page_mask)
    cv2.imwrite(str(debug_dir / "4_content_mask.jpg"), content_mask)
    
    # Calculate local density with larger kernel
    kernel_size = 99  # Changed to odd number
    density = cv2.GaussianBlur(content_mask.astype(float), (kernel_size, kernel_size), 0)
    
    # Normalize density
    density = (density / density.max() * 255).astype(np.uint8)
    cv2.imwrite(str(debug_dir / "5_density.jpg"), density)
    
    # Create edge mask
    edge_kernel = np.ones((25, 25), np.uint8)
    edge_mask = cv2.morphologyEx(page_mask, cv2.MORPH_GRADIENT, edge_kernel)
    cv2.imwrite(str(debug_dir / "6_edge_mask.jpg"), edge_mask)
    
    # Threshold density more aggressively
    _, damage_binary = cv2.threshold(density, 20, 255, cv2.THRESH_BINARY_INV)
    damage_binary = cv2.bitwise_and(damage_binary, page_mask)
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
            # Get contour center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Check if point is near edge
                if edge_mask[cy, cx] > 0 or np.any(edge_mask[max(0,cy-20):min(h,cy+20), max(0,cx-20):min(w,cx+20)] > 0):
                    hull = cv2.convexHull(contour)
                    cv2.drawContours(final_mask, [hull], -1, (255), -1)
                    cv2.drawContours(overlay, [hull], -1, (0, 0, 255), -1)
    
    # Final cleanup
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

if __name__ == "__main__":
    process_folder()