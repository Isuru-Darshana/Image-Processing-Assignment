import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 

im1 = cv.imread('assignment_01_images/spider.png')
im2 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)

hsv_image = cv.cvtColor(im2, cv.COLOR_BGR2HSV)
h,s,v = cv.split(hsv_image)

a = 0.5
sigma = 70
transformed_s = np.minimum(s+a*128*np.exp(-(s-128)**2/(2*sigma**2)),255).astype(np.uint8)
im_hsv_transformed = cv.merge([h,transformed_s,v])
im_transformed = cv.cvtColor(im_hsv_transformed, cv.COLOR_HSV2BGR)

Images = [im2, transformed_s,
          im_hsv_transformed, im_transformed]

titles = ['Original Image', 'Saturation Transformed',
          'HSV Merged Transform', 'Tranformed Image']

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(Images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.savefig('Q3/SaturationPlane.jpg')
plt.show()
