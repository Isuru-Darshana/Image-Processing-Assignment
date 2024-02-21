import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im_org = cv.imread('assignment_01_images/highlights_and_shadows.jpg')

img_lab = cv.cvtColor(im_org, cv.COLOR_BGR2LAB)
L, a, b = cv.split(img_lab)

gamma = 2.5
table = np.array([(i / 255) ** (1/gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
L_gamma = cv.LUT(L, table)

img_lab_gamma = cv.merge([L_gamma, a, b])
img_gamma = cv.cvtColor(img_lab_gamma, cv.COLOR_LAB2BGR)

fig, axarr = plt.subplots(3, 2, figsize=(12, 8))
axarr[0, 0].imshow(cv.cvtColor(im_org, cv.COLOR_BGR2RGB))
axarr[0, 0].set_title('Original Image')
axarr[0, 1].imshow(cv.cvtColor(img_gamma, cv.COLOR_BGR2RGB))
axarr[0, 1].set_title('Gamma Corrected Image')

color = ('b', 'g', 'r')
hist_org = [cv.calcHist([im_org], [i], None, [256], [0, 256]) for i in range(3)]
hist_gamma = [cv.calcHist([img_gamma], [i], None, [256], [0, 256]) for i in range(3)]

gamma = 1.5
gamma_transform = np.array([((i / 255.0) **(1/ gamma)) * 255 for i in np.arange(256)]).astype('uint8')

fig, axarr = plt.subplots(2, 3, figsize=(18, 10))  
axarr[0, 0].imshow(cv.cvtColor(im_org, cv.COLOR_BGR2RGB))
axarr[0, 0].set_title('Original Image')
axarr[0, 1].imshow(cv.cvtColor(img_gamma, cv.COLOR_BGR2RGB))
axarr[0, 1].set_title('1/Gamma Corrected Image')
axarr[0, 2].axis('off')
axarr[1, 0].plot(gamma_transform, color='black')
axarr[1, 0].set_title('Gamma Transformation Curve')

for i, col in enumerate(color):
    axarr[1, 1].plot(hist_org[i], color=col)
    axarr[1, 1].set_title('Histogram for Original Image')

for i, col in enumerate(color):
    axarr[1, 2].plot(hist_gamma[i], color=col)
    axarr[1, 2].set_title('Histogram for Gamma Corrected Image')

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig("Q2/Gamma Correction_OVER.png")
plt.show()