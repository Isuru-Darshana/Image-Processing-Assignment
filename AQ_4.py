import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

# Ensure the correct image path
im = cv.imread('assignment_01_images/shells.tif', 0) # Corrected path

# Compute histogram and CDF
hist, bins = np.histogram(im.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Equalize the histogram of the grayscale image
equ = cv.equalizeHist(im)

# Creating subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Original Image
axes[0].imshow(im, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Equalized Image
axes[1].imshow(equ, cmap='gray')
axes[1].set_title('Vibrance-Enhanced Image')
axes[1].axis('off')

axes[2].hist(im.flatten(), 256, [0, 256], color='b', label='Histogram')
axes[2].set_xlim([0, 256])
axes[2].legend(loc='upper left')
axes[2].set_title('Histogram')

plt.tight_layout()
plt.savefig('Q4/Histogram.jpg')
plt.show()
