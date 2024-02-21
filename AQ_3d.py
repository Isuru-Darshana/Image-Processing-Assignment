import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

im1 = cv.imread('assignment_01_images/spider.png')

hsv_image = cv.cvtColor(im1 , cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv_image)

a = 0.5
sigma = 70

transformed_s = np.minimum(s + a * 128 * np.exp(-(s - 128) ** 2 / (2 * sigma ** 2)), 255).astype(np.uint8)

im_transformed = cv.merge([h, transformed_s, v])
im_hsv_transformed = cv.cvtColor(im_transformed, cv.COLOR_HSV2BGR)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv.cvtColor(im1, cv.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cv.cvtColor(im_hsv_transformed, cv.COLOR_BGR2RGB))
axes[1].set_title('Vibrance Enhanced Image')
axes[1].axis('off')

x = np.arange(0, 256)
transformation = np.minimum(x + a * 128 * np.exp(-(x - 128) ** 2 / (2 * sigma ** 2)), 255)
axes[2].plot(x, transformation, color='blue')
axes[2].set_title('Intensity Transformation')
axes[2].set_xlabel('Input Intensity')
axes[2].set_ylabel('Output Intensity')
axes[2].set_xlim([0, 255])
axes[2].set_ylim([0, 255])

plt.tight_layout()
plt.savefig('Q3/SpiderIntensity.jpg')
plt.show()
