import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('assignment_01_images/einstein.png',cv.IMREAD_GRAYSCALE)

kernel = np.array([(1, 0, -1),(2, 0, -2),(1, 0, -1)], dtype= 'float')
img1 = cv.filter2D(img, -1, kernel)
fig, axes = plt.subplots(1, 2, sharex='all', sharey= 'all', figsize=(5,5))
axes[0].imshow(img,cmap = 'gray')
axes[0].set_title('Original')
axes[0].set_xticks([]),axes[0].set_yticks([])
axes[1].imshow(img1,cmap = 'gray')
axes[1].set_title('Sobel Horizontal')
axes[1].set_xticks([]),axes[1].set_yticks([])
plt.savefig('Q6/partb.png')
plt.show()