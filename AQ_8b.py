import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('assignment_01_images/daisy.jpg')
mask = cv.imread('Q8/updated_mask.png', 0)

if img is not None and mask is not None:
    
    mask_bool = mask.astype(bool)
    mask_3d = mask_bool[:, :, np.newaxis]

    
    blurred_img = cv.GaussianBlur(img, (21, 21), 0)

    enhanced_img = img.copy()
    enhanced_img[~mask_3d] = blurred_img[~mask_3d]

   
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv.cvtColor(enhanced_img, cv.COLOR_BGR2RGB))
    plt.title('Enhanced Image with Blurred Background')
    plt.axis('off')

    plt.tight_layout()
    plt.show()