import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('assignment_01_images/daisy.jpg')


if img is None:
    print('Error: Image cannot be loaded! Please check the path!')
else:
    
    rect = (30, 150, 539, 400) 

   
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    img_foreground = img * mask2[:, :, np.newaxis]
    img_background = img * (1 - mask2)[:, :, np.newaxis]

    
    plt.figure(figsize=(10, 5))

    plt.subplot(131)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask2, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv.cvtColor(img_foreground, cv.COLOR_BGR2RGB))
    plt.title('Foreground Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('Q8/parta.png')
    plt.show()

    cv.imwrite('Q8/updated_mask.png', mask2 * 255)
    cv.imwrite('Q8/updated_foreground.png', img_foreground)
    cv.imwrite('Q8/updated_background.png', img_background)
    