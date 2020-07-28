import cv2
import numpy as np

threshold1 = 200
threshold2 = 90

# https://stackoverflow.com/a/45579542/879243
def processImage(image):
    processed = processImage(image)
    return processed

def processImage(image):
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # adaptiveThreshold accepts single channel (grayscale) images
    # 200 works well for the threshold
    at = cv2.adaptiveThreshold(gray, threshold1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)

    # Otsu's binarization
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #data,otsu = cv2.threshold(blur,threshold1,threshold2,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Masking with the thresholds. The two arrays must match, so expand thresholds to RGB
    mask = cv2.cvtColor(at, cv2.COLOR_GRAY2RGB)
    masked = cv2.bitwise_and(image, mask)

    # Get the background. Blur it an desaturate it
    backgroundMask = cv2.bitwise_not(mask)
    backgroundBlur = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)

    # Combine background with foreground
    composite = cv2.bitwise_or(masked, backgroundBlur)

    return composite

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
