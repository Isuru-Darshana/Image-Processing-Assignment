import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt

im = cv.imread('assignment_01_images/rice_gaussian_noise.png')
noise_removed_im = cv.fastNlMeansDenoising(im, None, 40,7,21)

fig,axarr = plt.subplots(2,2)
axarr[0,0].imshow(im)
axarr[0,0].set_title('Original Image')
axarr[0,1].imshow(noise_removed_im)
axarr[0,1].set_title('Noise Removed Image')
 
images = [im, noise_removed_im]
titles = ['Original Image','Noise Removed Image']

plt.figure(figsize = (10, 8))

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.savefig('Q5/part_a.png')
plt.show()