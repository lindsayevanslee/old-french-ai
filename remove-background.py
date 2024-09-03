import cv2
import numpy as np
import os

def black_out_background(image_path, save_intermediates=False):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Get the directory of the input file
    output_dir = os.path.dirname(image_path)

    def save_step(name, img):
        if save_intermediates:
            cv2.imwrite(os.path.join(output_dir, f"{name}.png"), img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_step("02_grayscale", gray)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_step("03_blurred", blurred)

    # Apply a binary threshold to the image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_step("04_thresholded", thresholded)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    contour_image = original.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    save_step("05_all_contours", contour_image)

    # Sort contours by area in descending order
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Select contours that are likely to be pages
    total_area = gray.shape[0] * gray.shape[1]
    page_contours = [cnt for cnt in sorted_contours if cv2.contourArea(cnt) > total_area * 0.1]

    # Draw the selected contours
    selected_contours_image = original.copy()
    cv2.drawContours(selected_contours_image, page_contours, -1, (0, 255, 0), 3)
    save_step("06_selected_contours", selected_contours_image)

    # Create a mask from the selected contours
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, page_contours, -1, (255), thickness=cv2.FILLED)
    save_step("07_mask", mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(original, original, mask=mask)
    save_step("08_masked_result", result)

    # Create a black background
    black_background = np.zeros_like(original)

    # Combine the result with the black background
    final_result = cv2.bitwise_or(result, black_background)
    
    # Always save the final result
    cv2.imwrite(os.path.join(output_dir, '09_background_removed.png'), final_result)

    if save_intermediates:
        print(f"Processed {image_path}. Intermediate outputs saved in {output_dir}")
    else:
        print(f"Processed {image_path}. Final result saved in {output_dir}")

    return final_result

def process_directory(input_directory, save_intermediates=False):
    for root, dirs, files in os.walk(input_directory):

        #skip the old directory
        if root.endswith("old"):
            continue

        print(f"root: {root}")
        print(f"dirs: {dirs}")
        print(f"files: {files}")

        if '01_resized.png' in files:
            image_path = os.path.join(root, '01_resized.png')
            print(f"Processing: {image_path}")
            black_out_background(image_path, save_intermediates)        
        elif '0_converted_to_png.png' in files:
            image_path = os.path.join(root, '0_converted_to_png.png')
            print(f"Processing: {image_path}")
            black_out_background(image_path, save_intermediates)
        else:
            print(f"No suitable image found in {root}")

# Usage example
input_directory = 'data/ME MSS Images/test output'
save_intermediates = True  # Set this to False to skip saving intermediate outputs

process_directory(input_directory, save_intermediates)

