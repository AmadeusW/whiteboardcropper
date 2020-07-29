from skimage import exposure
import numpy as np
import imutils
import cv2
import math 

class ImageEnhancer:

    def enhance(self, image):
        # See clarity.py
        # Somehow we're getting float64 image. convert it to something that cvtColor can understand
        image = (image*255).astype('uint8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # adaptiveThreshold accepts single channel (grayscale) images
        # Good values are 198 for the threshold, and a low blockSize, e.g. 3
        at = cv2.adaptiveThreshold(gray, 198, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,3,2)

        foreground = cv2.bitwise_and(image, image, mask=at)

        # Get the background. Blur it an desaturate it
        #backgroundBlur = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
        #backgroundBlur = cv2.bilateralFilter(image, 5, TestValue, TestValue)
        backgroundBlur = image # don't do any smoothing of the background

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