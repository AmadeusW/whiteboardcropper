import cv2
import numpy as np

threshold1 = 198
threshold2 = 3

# https://stackoverflow.com/a/45579542/879243
def processImage(image):
    processed = processImage(image)
    return processed

def processImage(image):
    # https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # adaptiveThreshold accepts single channel (grayscale) images
    # Good values are 198 for the threshold, and a low blockSize, e.g. 3
    at = cv2.adaptiveThreshold(gray, 198, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)
    # Alternatively, use edge detection
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #edges = cv2.Canny(gray, threshold1, threshold2, 3) # 10, 90

    # Otsu's binarization
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    #data,otsu = cv2.threshold(blur,threshold1,threshold2,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    foreground = cv2.bitwise_and(image, image, mask=at)

    # Get the background. Blur it an desaturate it
    backgroundBlur = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    amplifiedLight = 0.9
    addedGamma = 0.5
    # Make the background brighter prior to multiplication
    # Values 0.0, 0.9 and 0.5 (=128) seem to do well
    washedOut = cv2.addWeighted(backgroundBlur, 0.0, backgroundBlur, amplifiedLight, addedGamma*255)
    # remove the area taken care of by the edge detector
    # I can't figure out why, but we can't use mask=bitwise_not(at) like above.
    # BTW we convert to BGR is that the two arrays for bitwise_and must match number of channels
    mask = cv2.cvtColor(at, cv2.COLOR_GRAY2BGR)
    backgroundMask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(washedOut, backgroundMask)

    # Combine background with foreground
    composite = cv2.add(background, foreground)
    return composite

# Helper methods
def increaseV1():
    global threshold1
    if threshold1 <= 250:
        threshold1 += 5
def decreaseV1():
    global threshold1
    if threshold1 >= 5:
        threshold1 -= 2
def increaseV2():
    global threshold2
    if threshold2 <= 31:
        threshold2 += 2
def decreaseV2():
    global threshold2
    if threshold2 >= 5:
        threshold2 -= 2

# Main code
# get the image from the video camera
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
#capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

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
