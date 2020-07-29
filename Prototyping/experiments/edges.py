import cv2
import numpy as np

threshold1 = 10
threshold2 = 90

# https://stackoverflow.com/a/45579542/879243
def processImage(image):
    edges = basicEdgeDetection(image)
    return edges

def basicEdgeDetection(image):
    # Basic edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    # Adjust the thresholds; Canny(blur_gray, low_threshold, high_threshold, apertureSize, bool L2gradient)
    # The lower the thresholds, the more lines and noise we see.
    # I wouldn't go lower than 30 for high threshold, and not higher than 100 where we stop seeing the whiteboard
    # The lower threshold seemed to be fine at 10
    edges = cv2.Canny(blur, threshold1, threshold2, 3) # 10, 90
    return edges

# Helper methods
def increaseV1():
    global threshold1
    if threshold1 < 250:
        threshold1 += 5
def decreaseV1():
    global threshold1
    if threshold1 > 5:
        threshold1 -= 5
def increaseV2():
    global threshold2
    if threshold2 < 250:
        threshold2 += 5
def decreaseV2():
    global threshold2
    if threshold2 > 5:
        threshold2 -= 5

# Main code
# get the image from the video camera
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    result = processImage(frame)

    diagnostics = '{},{}'.format(threshold1, threshold2)
    cv2.putText(result, diagnostics, (5,20), cv2.FONT_HERSHEY_PLAIN,1, (200, 255, 200, 255), 1)

    cv2.imshow('Whiteboard', result)
    key = cv2.waitKey(1)
    if key == 27: # Escape
        break
    elif key == 119: # w
        increaseV1()
    elif key == 115: # S
        decreaseV1()
    elif key == 100: # d
        increaseV2()
    elif key == 97: # a
        decreaseV2()

# release resources
capture.release()
cv2.destroyAllWindows()
