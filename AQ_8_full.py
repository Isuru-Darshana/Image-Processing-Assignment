import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image_path = 'assignment_01_images/daisy.jpg'
image = cv.imread(image_path)

# Check if the image is loaded correctly
if image is not None:
    # Define the initial rectangle parameters
    # This rectangle should cover the flower
    rect = (30, 150, 539, 400)

    # Initialize mask, bgdModel, fgdModel for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

    #
    mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')

    
    segmented = image * mask2[:, :, np.newaxis]

   
    blurred_background = cv.GaussianBlur(image, (21, 21), 0)

    
    enhanced_image = blurred_background * (1 - mask2[:, :, np.newaxis]) + segmented
    background_image = blurred_background * (1 - mask2[:, :, np.newaxis])

    
    plt.figure(figsize=(15, 10))
    plt.subplot(231), plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(232), plt.imshow(mask2, 'gray'), plt.title('Segmentation Mask')
    plt.subplot(233), plt.imshow(cv.cvtColor(segmented, cv.COLOR_BGR2RGB)), plt.title('Foreground Image')
    plt.subplot(234), plt.imshow(cv.cvtColor(background_image, cv.COLOR_BGR2RGB)), plt.title('Background Image')
    plt.subplot(235), plt.imshow(cv.cvtColor(enhanced_image, cv.COLOR_BGR2RGB)), plt.title('Enhanced Image with Blurred Background')
    plt.tight_layout()
    plt.show()