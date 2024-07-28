import cv2
import numpy as np
import os

def black_out_non_book_areas(image_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    original = image.copy()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to the image
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, which should be the book
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Invert the mask to black out non-book areas
    mask = cv2.bitwise_not(mask)

    # Black out non-book areas in the original image
    result = cv2.bitwise_and(original, mask)

    # Save the result
    cv2.imwrite(output_path, result)

    # Display the result (optional)
    """
    cv2.imshow('Original Image', original)
    cv2.imshow('Processed Image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    
# Usage example
input_directory = 'data/ME MSS Images/test images'
output_directory = 'data/ME MSS Images/test output'

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    # Check if the file is an image
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Construct the full path of the input and output files
        image_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)

        print(f"Processing: {image_path}; saving to: {output_path}")
        
        # Call the function to black out non-book areas
        black_out_non_book_areas(image_path, output_path)
