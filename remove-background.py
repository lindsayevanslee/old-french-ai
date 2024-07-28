import cv2
import numpy as np
import os

def black_out_background(image_path, output_dir, save_intermediates=False):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Create a subdirectory for this image's output
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)

    def save_step(name, img):
        if save_intermediates:
            cv2.imwrite(os.path.join(image_output_dir, f"{name}.jpg"), img)

    save_step("1_original", original)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_step("2_grayscale", gray)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    save_step("3_blurred", blurred)

    # Apply a binary threshold to the image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save_step("4_thresholded", thresholded)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    contour_image = original.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
    save_step("5_all_contours", contour_image)

    # Find the largest contour, which should be the book
    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour
    largest_contour_image = original.copy()
    cv2.drawContours(largest_contour_image, [largest_contour], -1, (0, 255, 0), 3)
    save_step("6_largest_contour", largest_contour_image)

    # Create a mask from the largest contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
    save_step("7_mask", mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(original, original, mask=mask)
    save_step("8_masked_result", result)

    # Create a black background
    black_background = np.zeros_like(original)

    # Combine the result with the black background
    final_result = cv2.bitwise_or(result, black_background)
    
    # Always save the final result
    cv2.imwrite(os.path.join(image_output_dir, 'final_result.jpg'), final_result)

    if save_intermediates:
        print(f"Processed {image_path}. Intermediate outputs saved in {image_output_dir}")
    else:
        print(f"Processed {image_path}. Final result saved in {image_output_dir}")

    return final_result

# Usage example
input_directory = 'data/ME MSS Images/test images'
output_directory = 'data/ME MSS Images/test output'
save_intermediates = False  # Set this to False to skip saving intermediate outputs

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct the full path of the input file
        image_path = os.path.join(input_directory, filename)
        print(f"Processing: {image_path}")
        # Call the function to black out the background and optionally save intermediate outputs
        black_out_background(image_path, output_directory, save_intermediates)