from skimage import exposure
import numpy as np
import imutils
import cv2
import math 

class WhiteBoardFinder:

    def __init__(self, precision, canny1, canny2):
        self.precision = precision
        self.canny1 = canny1
        self.canny2 = canny2
        self.biggestAreaContours = None
        self.largestAreaFound = 1000
    
    def resetCache(self):
        self.biggestAreaContours = None
        self.largestAreaFound = 1000

    def findBoard(self, image):
        # compute the ratio of the old height
        # to the new height, clone it, and resize it
        ratio = image.shape[0] / 300.0
        orig = image.copy()
        image = imutils.resize(image, height = 300)
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 17, 17)
        edged = cv2.Canny(gray, self.canny1, self.canny2)

        # FIND THE SCREEN
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None
        area = None

        # loop over our contours WHICH ARE NOW SORTED IN ORDER FROM BIGGEST DOWN
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            area = cv2.contourArea(c)

            # There is no way a tiny board can happen so limit min size
            if (area < self.largestAreaFound-100):
                continue
            
            # If it is the largest go with it
            elif (self.biggestAreaContours is not None and area == self.largestAreaFound):
                screenCnt = BiggestAreaContours
                break

            approxOutline = cv2.approxPolyDP(c, self.precision * peri, True)

            # if our approximated contour has four or more points, then
            # we can assume that we have found our screen and can do further
            # processing if needed
            if len(approxOutline) >= 4:
                
                # if we made it this far it is a good contour and so we can save it
                LargestAreaFound = area
                BiggestAreaContours = approxOutline

                screenCnt = approxOutline
                break

        if (screenCnt is None):
            cv2.putText(image, "Whiteboard lost. Please press r to reset", (5, 25), cv2.FONT_HERSHEY_COMPLEX , 0.5, (0, 255, 0))  
            screenCnt = self.biggestAreaContours

        return screenCnt, ratio, image, edged, area