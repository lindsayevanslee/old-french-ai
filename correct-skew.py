# Description: This script corrects the skew of the image by rotating it to the best angle.
# adapted from: https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
#from scipy.ndimage import interpolation as inter
from scipy.ndimage import rotate
from tqdm import tqdm

#set the input directory
input_dir = 'data/ME MSS Images/test output' 

#set the delta and limit for the rotation, in degrees
delta = 1
limit = 15

# Function to find the best angle to rotate the image to correct skew
def find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

# Loop through all files recursively in the input directory
for root, dirs, files in os.walk(input_dir):
    
    #skip the old directory
    if root.endswith("old"):
        continue

    print(f"root: {root}")
    print(f"dirs: {dirs}")

    for filename in tqdm(files):

        #if filename == "0_converted_to_png.png":
        if filename == "09_background_removed.png":

            #create path of input image
            input_file = os.path.join(root, filename)
            print(f"Converting: {input_file}")

            #create path of output binary image
            #new_root_binary = root.replace('png', 'binary')
            #os.makedirs(new_root_binary, exist_ok=True)
            #output_file_binary = os.path.join(new_root_binary, filename)
            output_file_binary = os.path.join(root, "10_binary.png")

            #create path of output skew corrected image
            #new_root_skew = root.replace('png', 'skew_corrected')
            #os.makedirs(new_root_skew, exist_ok=True)
            #output_file_skew = os.path.join(new_root_skew, filename)
            output_file_skew = os.path.join(root, "11_skew_corrected.png")

            #check if skew corrected image already exists. If so, skip iteration (we assume the binary file also exists, so skipping that too)
            if os.path.exists(output_file_skew):
                print(f"File already exists: {output_file_skew}")
                continue
            else: 
                # read the image
                img = im.open(input_file)

                # convert to binary
                wd, ht = img.size
                try:
                    pix = np.array(img.convert('1').getdata(), np.uint8)
                except OSError:
                    print(f"Failed to process image: {input_file}. The file may be truncated or corrupted.")
                    continue
                bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)

                #check if binary image already exists
                if os.path.exists(output_file_binary):
                    print(f"File already exists: {output_file_binary}")
                    
                else:
                    print(f"Printing binary file: {input_file} to {output_file_binary}")

                    plt.imshow(bin_img, cmap='gray')
                    plt.savefig(output_file_binary)


                print(f"Correcting skew: {output_file_binary} to {output_file_skew}")

                # find best angle
                angles = np.arange(-limit, limit+delta, delta)
                scores = []

                for angle in angles:
                    hist, score = find_score(bin_img, angle)
                    scores.append(score)

                best_score = max(scores)
                best_angle = angles[scores.index(best_score)]
                print('Best angle: {}'.format(best_angle))
                # save the angle of the corrected skew in a text file
                angle_file = os.path.join(root, "skew_angle.txt")
                with open(angle_file, 'w') as f:
                    f.write(str(best_angle))


                # correct skew
                data = rotate(bin_img, best_angle, reshape=False, order=0)
                img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
                img.save(output_file_skew)
            
        else:
            continue