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
    
    # Get content mask (text and other dark elements)
    _, content_binary = cv2.threshold(gray, background_color - 20, 255, cv2.THRESH_BINARY_INV)
    content_binary = cv2.bitwise_and(content_binary, page_mask)
    
    # Dilate the content to connect nearby elements
    kernel = np.ones((5,5), np.uint8)
    content_dilated = cv2.dilate(content_binary, kernel, iterations=2)
    cv2.imwrite(str(debug_dir / "4_content_dilated.jpg"), content_dilated)
    
    # The damage mask is areas within the page but not near content
    damage_mask = cv2.bitwise_and(page_mask, cv2.bitwise_not(content_dilated))
    
    # Clean up the damage mask
    kernel = np.ones((3,3), np.uint8)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(str(debug_dir / "5_damage_mask_initial.jpg"), damage_mask)
    
    # Find and filter damage contours
    damage_contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create final mask and overlay
    final_mask = np.zeros_like(gray)
    overlay = img.copy()
    
    for contour in damage_contours:
        area = cv2.contourArea(contour)
        if area > min_hole_area:
            # Get the average intensity in this region
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, (255), -1)
            region = cv2.bitwise_and(gray, mask)
            avg_intensity = cv2.mean(gray, mask=mask)[0]
            
            # Only include if the intensity is close to background color
            if abs(avg_intensity - background_color) < 30:
                cv2.drawContours(final_mask, [contour], -1, (255), -1)
                cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
    
    # Create final overlay
    alpha = 0.5
    overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # Save final outputs
    cv2.imwrite(str(debug_dir / "6_final_mask.jpg"), final_mask)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    
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