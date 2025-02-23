import cv2
import numpy as np
import os
from pathlib import Path

def generate_damage_mask(image_path, output_path, overlay_path, min_hole_area=10000):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    
    # Remove small noise
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Create empty mask
    mask = np.zeros_like(gray)
    
    # Create overlay image
    overlay = img.copy()
    
    # Draw only large contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_hole_area:
            # Draw on mask
            cv2.drawContours(mask, [contour], -1, (255), -1)
            
            # Draw on overlay in red
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
    
    # Blend overlay with original image
    alpha = 0.5  # Transparency factor
    overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    
    # Draw contour boundaries in bright red
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_hole_area:
            cv2.drawContours(overlay_img, [contour], -1, (0, 0, 255), 2)
    
    # Create output directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the mask and overlay
    cv2.imwrite(str(output_path), mask)
    cv2.imwrite(str(overlay_path), overlay_img)
    
    print(f"Saved mask to: {output_path}")
    print(f"Saved overlay to: {overlay_path}")
    
    return mask, overlay_img

def process_folder():
    # Set up input and output paths
    base_path = Path("data/digitized versions/Estoire du Graal - Merlin en prose -Suite Vulgate")
    input_dir = base_path / "jpeg"
    mask_dir = base_path / "masks"
    overlay_dir = base_path / "overlays"
    
    # Create output directories
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for images in: {input_dir}")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
        
    # Process all jpeg files
    for img_path in input_dir.glob("*.jpeg"):
        # Generate output paths
        mask_path = mask_dir / f"{img_path.stem}_mask.jpeg"
        overlay_path = overlay_dir / f"{img_path.stem}_overlay.jpeg"
        
        print(f"\nProcessing {img_path}")
        try:
            mask, overlay = generate_damage_mask(
                img_path, 
                mask_path, 
                overlay_path
            )
            if mask is None:
                print(f"Failed to process {img_path}")
            else:
                print(f"Successfully processed {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    process_folder()