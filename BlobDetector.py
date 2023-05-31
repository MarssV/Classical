import cv2
import numpy as np
from matplotlib import pyplot as plt


def detect_scratches(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a blob detector with the desired parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set the thresholding parameters
    params.minThreshold = 10
    params.maxThreshold = 500

    # Filter by area to remove small blobs
    params.filterByArea = True
    params.minArea = 100

    # Filter by circularity to remove non-circular blobs
    params.filterByCircularity = True
    params.minCircularity = 0.07

    # Filter by convexity to remove non-convex blobs
    params.filterByConvexity = True
    params.minConvexity = 0.08

    # Create the blob detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the grayscale image
    keypoints = detector.detect(gray)

    # Draw detected blobs on the original image
    result = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the result
    cv2.imshow("Scratch Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the image
image = cv2.imread("car.png")

# Perform scratch detection
detect_scratches(image)
