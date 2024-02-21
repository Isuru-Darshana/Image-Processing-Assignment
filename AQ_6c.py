import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

im1 = cv.imread('assignment_01_images/einstein.png',cv.IMREAD_GRAYSCALE)

Px = np.array([[1],[2],[1]])@np.array([[1, 0, -1]])
Py = np.array([[1, 2, 1]])@np.array([[1],[0],[-1]])

Gx = cv.filter2D(im1, -1, Px)
Gy = cv.filter2D(im1, -1, Py)

gradient_magnitude = np.sqrt(Gx**2+Gy**2)

plt.figure(figsize = (10,5))
plt.subplot(1, 3, 1)
plt.imshow(im1, cmap = 'gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap = 'gray')
plt.title('Gradient Magnitude')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(Gx, cmap = 'gray')
plt.title('Gx (Horizontal Edges)')
plt.axis('off')
plt.savefig('Q6/partc.png')
plt.show()