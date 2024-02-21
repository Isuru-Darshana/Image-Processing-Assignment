import cv2 as cv
from matplotlib import pyplot as plt

def zoom_image(image, scale_factor):
   
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    
    zoomed_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_NEAREST)
    return zoomed_image


image_path = "assignment_01_images/a1q5images/im01.png"
image = cv.imread(image_path, cv.IMREAD_COLOR)


if image is None:
    print("Error: Unable to load image.")
else:
    
    scale_factor = 3.0

    
    if scale_factor <= 0 or scale_factor > 10:
        print("Error: Invalid scale factor. Scale factor must be in the range (0, 10].")
    else:
       
        zoomed_image = zoom_image(image, scale_factor)
        original_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        zoomed_rgb = cv.cvtColor(zoomed_image, cv.COLOR_BGR2RGB)
        
       
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(original_rgb)
        ax[0].set_title('Original')
        ax[1].imshow(zoomed_rgb)
        ax[1].set_title('Zoomed')
        plt.show()

