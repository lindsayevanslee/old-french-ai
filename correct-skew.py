# Description: This script corrects the skew of the image by rotating it to the best angle.
# adapted from: https://towardsdatascience.com/pre-processing-in-ocr-fc231c6035a7

import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
#from scipy.ndimage import interpolation as inter
from scipy.ndimage import rotate


#input_file = sys.argv[1]
#example file
input_file = 'data/ME MSS Images/png/L1-B/IMG_5658.png'
output_file_binary = 'data/ME MSS Images/binary/L1-B/IMG_5658.png'
output_file_skew = 'data/ME MSS Images/skew_corrected/L1-B/IMG_5658.png'

img = im.open(input_file)
# convert to binary
wd, ht = img.size
pix = np.array(img.convert('1').getdata(), np.uint8)
bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
plt.imshow(bin_img, cmap='gray')
plt.savefig(output_file_binary)
def find_score(arr, angle):
    #data = inter.rotate(arr, angle, reshape=False, order=0)
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score
delta = 1
limit = 15
angles = np.arange(-limit, limit+delta, delta)
scores = []
for angle in angles:
    hist, score = find_score(bin_img, angle)
    scores.append(score)
best_score = max(scores)
best_angle = angles[scores.index(best_score)]
print('Best angle: {}'.format(best_angle))
# correct skew
#data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
data = rotate(bin_img, best_angle, reshape=False, order=0)
img = im.fromarray((255 * data).astype("uint8")).convert("RGB")
img.save(output_file_skew)