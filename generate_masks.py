import cv2
import numpy as np
import os
from pathlib import Path

def generate_damage_mask(image_path, output_path, overlay_path, debug_dir, min_hole_area=3000):
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
    
    # Apply local adaptive threshold to get text and content
    adaptive_thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,  # Block size - reduced from original
        5    # C constant - increased from original
    )
    
    # Apply page mask
    content_mask = cv2.bitwise_and(adaptive_thresh, page_mask)
    cv2.imwrite(str(debug_dir / "4_content_mask.jpg"), content_mask)
    
    # Calculate local density of content with smaller kernel
    kernel_size = 25  # Reduced from 50
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    density = cv2.blur(content_mask, (kernel_size, kernel_size))
    cv2.imwrite(str(debug_dir / "5_density.jpg"), density)
    
    # Threshold density with higher threshold
    mean_density = np.mean(density[page_mask > 0])
    _, damage_binary = cv2.threshold(density, mean_density * 0.5, 255, cv2.THRESH_BINARY_INV)  # Increased from 0.3
    damage_binary = cv2.bitwise_and(damage_binary, page_mask)
    
    # More aggressive cleanup of the damage mask
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((7,7), np.uint8)
    damage_binary = cv2.morphologyEx(damage_binary, cv2.MORPH_OPEN, kernel_small)
    damage_binary = cv2.morphologyEx(damage_binary, cv2.MORPH_CLOSE, kernel_medium)
    damage_binary = cv2.morphologyEx(damage_binary, cv2.MORPH_OPEN, kernel_medium)
    cv2.imwrite(str(debug_dir / "6_damage_binary.jpg"), damage_binary)
    
    # Find and filter damage contours
    damage_contours, _ = cv2.findContours(damage_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask and overlay
    final_mask = np.zeros_like(gray)
    overlay = img.copy()
    
    for contour in damage_contours:
        area = cv2.contourArea(contour)
        if area > min_hole_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Relaxed aspect ratio constraints
            if 0.1 < aspect_ratio < 10:  # Wider range than before
                cv2.drawContours(final_mask, [contour], -1, (255), -1)
                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
    
    # Final cleanup of the mask
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Create final overlay
    alpha = 0.5
    overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # Save outputs
    cv2.imwrite(str(debug_dir / "7_final_mask.jpg"), final_mask)
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
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
        
    for img_path in input_dir.glob("*.jpeg"):
        print(f"\nProcessing {img_path}")
        
        # Create debug directory for this image
        img_debug_dir = debug_dir / img_path.stem
        mask_path = mask_dir / f"{img_path.stem}_mask.jpeg"
        overlay_path = overlay_dir / f"{img_path.stem}_overlay.jpeg"
        
        try:
            mask, overlay = generate_damage_mask(
                img_path, 
                mask_path, 
                overlay_path,
                img_debug_dir
            )
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    process_folder()