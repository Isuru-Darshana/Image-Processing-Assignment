import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

im1 = cv.imread("assignment_01_images/rice_gaussian_noise.png", cv.IMREAD_GRAYSCALE)

im2 = cv.imread("assignment_01_images/rice_salt_pepper_noise.png", cv.IMREAD_GRAYSCALE)

if im1 is None or im2 is None:
    print("Error: Unable to read one or both images")
else:
    print("Images were read successfully")

retVal1, thresh_otsu1 = cv.threshold(im1, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

retVal2, thresh_otsu2 = cv.threshold(im2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

images = [im1, thresh_otsu1,
          im2, thresh_otsu2]

titles = ['Original Noisy Image 1', "Otsu's Thresholding",
          'Original Noisy Image 2', "Otsu's Thresholding"]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig("Q5/partd.png")
plt.show()
