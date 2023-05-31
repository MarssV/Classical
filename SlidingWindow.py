import cv2
from matplotlib import pyplot as plt
import numpy as np



def sliding_window(image, window_size, stride):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Slide the window across the image
    for y in range(0, height - window_size[1] + 1, stride):
        for x in range(0, width - window_size[0] + 1, stride):
            # Extract the current window
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Apply Canny edge detection on the window
            edges = cv2.Canny(window, 100, 200)
            hist = cv2.calcHist([edges],[0],None,[256],[0,256])
            if hist[0]<2200:
                 mask[y:y + window_size[1], x:x + window_size[0]] = 255
                 

##            plt.plot(hist)
##            plt.xlim([0,256])
##            plt.show()



            # Perform further processing on the edges
            # ...

            # Display the window and edges
            cv2.imshow("Window", window)
            cv2.imshow("Edges", edges)
            cv2.waitKey(100)

    cv2.destroyAllWindows()
    cv2.imshow("mask", mask)



# Load the image
image1 = cv2.imread("car2.PNG")

image = cv2.imread("car2.png", cv2.IMREAD_GRAYSCALE)
mask = np.zeros(image.shape[:2], np.uint8)


# Define the window size and stride
window_size = (50, 50)
stride = 50

# Perform sliding window and edge detection
sliding_window(image, window_size, stride)
masked_img = cv2.bitwise_and(image1,image1,mask = mask)

cv2.imshow("masked", masked_img)




