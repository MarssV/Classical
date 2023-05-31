#https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18

import cv2
import numpy as np
#from PIL import Image



def count_child_contours(contour, hierarchy):
    # Initialize the count variable
    count = 0
    hierarchy_info = hierarchy[0]

##  for j in range(0,len(contour),1):
##
##      child_count = np.count_nonzero(hierarchy_info[:, 3] == j)
        
    #print(child_count)
    #print(hierarchy_info[:, 3]) 


    # Iterate through the contour hierarchy
    
    for h in hierarchy[0]:
        if h[3] != -1:
            count += 1

    print(count)
    return count



# Load the image
image = cv2.imread("car2.png")
output = image.copy()


# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform thresholding
_, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(gray, 100, 200)
#threshold= threshold +edges


# Find contours and hierarchy in the thresholded image
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

mask = np.zeros_like(gray)


# Iterate through the contours
for i, contour in enumerate(contours):

    mask_int = np.zeros(image.shape[:2], np.uint8)

    x, y, w, h = cv2.boundingRect(contour)

##  if 500> w > 10 and 300> h > 10:

    mask_int[y:y + h, x:x + w] = 1


    masked_img = cv2.bitwise_and(threshold, threshold, mask=mask_int)
    #imgMasked = Image.fromarray(masked_img)

    # Check if the contour has more than three child contours
    _,hierarchy = cv2.findContours(masked_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

##    hierarchy1 = hierarchy[0][i][2]
##    print(hierarchy1)


    if 20> count_child_contours(contour, hierarchy) > 3:
        # Create a mask for the isolated contour region
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        cv2.drawContours(mask, contours, i, 255, -1)

        #cv2.drawContours(mask, contours, i, color, 1)
        
        cv2.drawContours(output, contour, -1, color , 1)

        # Apply the mask to isolate the contour region
        isolated_contour = cv2.bitwise_and(image, image, mask=mask)

        # Display the isolated contour region

cv2.imshow("Filtered Contours", output)

cv2.imshow("masked_img", masked_img)

cv2.imshow("Isolated Contour", isolated_contour)

cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()
