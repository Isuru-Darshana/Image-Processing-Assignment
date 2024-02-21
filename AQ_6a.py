from scipy import datasets,ndimage
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

img = cv.imread('assignment_01_images/einstein.png',cv.IMREAD_GRAYSCALE)

ascent = datasets.ascent().astype('int32')
sobel_h = ndimage.sobel(img, axis=0)
sobel_v = ndimage.sobel(img, axis=1)
magnitude = np.sqrt(sobel_h**2+sobel_v**2)
magnitude *= 255.0/np.max(magnitude)

fig, axs= plt.subplots(2,2, figsize= (8,8))
plt.gray()
axs[0,0].imshow(img)
axs[0,1].imshow(sobel_h)
axs[1,0].imshow(sobel_v)
axs[1,1].imshow(magnitude)
titles = ["Original","Horizontal","Vertical","Magnitude"]

for i,ax in enumerate(axs.ravel()):
    ax.set_title(titles[i])
    ax.axis('off')
plt.savefig('Q6/part_a')
plt.show()