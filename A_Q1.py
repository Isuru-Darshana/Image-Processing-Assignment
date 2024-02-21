import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

im1 = cv.imread("assignment_01_images/margot_golden_gray.jpg")
assert im1 is not None

t = np.zeros(256, dtype=np.uint8)
#t[0:50] = np.array([int(x*200/255) for x in range(50)])
#t[50:151] = np.array([int(x*200/255 + 50) for x in range(50,151)])
#t[151:256] = np.array([int(x*200/255) for x in range(151,256)])
t[0:221] = np.array([int(x*200/255) for x in range(221)])
t[221:256] = np.array([int(x*200/255 + 40) for x in range(221,256)])
im2 = t[im1]

fig, ax = plt.subplots(1,3, figsize=(15, 5))
ax[0].imshow(im1, vmin=0, vmax=255, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(im2, vmin=0, vmax=255, cmap='gray')
ax[1].set_title('Intensity Transformed')
ax[2].plot(t, color='maroon')
ax[2].set_title('(a) Intensity transformation. ')
ax[2].set_xlabel('Input Intensity')
ax[2].set_ylabel('Output Intensity')
ax[2].grid(True)
ax[2].set_xlim([0,255])
ax[2].set_ylim([0,255])

plt.tight_layout()
plt.savefig("Q1/Intensity transform.png")
plt.show()


