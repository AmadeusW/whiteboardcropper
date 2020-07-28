import cv2
import numpy as np

# https://stackoverflow.com/a/45579542/879243
def processImage(image):
    edges = basicEdgeDetection(image)
    return edges

def basicEdgeDetection(image):
    # Basic edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    # Adjust the thresholds; Canny(blur_gray, low_threshold, high_threshold, apertureSize, bool L2gradient)
    # The higher the high threshold, the more noise you see. So far, 50 seems like a good number
    edges = cv2.Canny(blur, 40, 50, 3)
    return edges

# get the image from the video camera
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    result = processImage(frame)
    cv2.imshow('Whiteboard Viewer', result)
    if cv2.waitKey(1) == 27: # Escape
        break

# release resources
capture.release()
cv2.destroyAllWindows()
