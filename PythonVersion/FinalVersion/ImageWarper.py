from skimage import exposure
import numpy as np
import imutils
import cv2
import math 

class ImageWarper:

    def __init__(self, whiteBoardFinder):
        self.whiteBoardFinder = whiteBoardFinder
        self.bestWarpFound = None

    def resetCache(self):
        self.biggestAreaContours = None
        self.largestAreaFound = None

    def getDistanceFromPoint(point, point2 = (0,0)):
        return math.sqrt((point[0] - point2[0])**2 + (point[1] - point2[1])**2)

    def orderPoints(points):
        sortedList = sorted(points, key=getDistanceFromPoint)
        target = np.array(sortedList)

        # We can't trust the top right and bottom left corners as they can flip. This is because we order based on distance from origin.
        # A slight rotation can therefore cause a difference and they will flip. However this is not the case with the top left and bottom
        # right corners. Compensate by making sure that the top right corner is the one with the larger x
        if (target[1][0] < target[2][0]):
            tmp = [target[1][0], target[1][1]] # done this way to pass actual value instead of reference
            target[1] = target[2]
            target[2] = tmp

        return target

    def warpPerspective(self, orig, screenCnt, resizeRatio, area):

        warp = None

        if (self.bestWarpFound is None or area != self.whiteBoardFinder.largestAreaFound):
            # now that we have our screen contour, we need to determine
            # the top-left, top-right, bottom-right, and bottom-left
            # points so that we can later warp the image -- we'll start
            # by reshaping our contour to be our finals and initializing
            # our output rectangle in top-left, top-right, bottom-right,
            # and bottom-left order
            pts = screenCnt.reshape(4, 2)
            rect = np.zeros((4, 2), dtype = "float32")
            # the top-left point has the smallest sum whereas the
            # bottom-right has the largest sum
            s = pts.sum(axis = 1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            # compute the difference between the points -- the top-right
            # will have the minumum difference and the bottom-left will
            # have the maximum difference
            diff = np.diff(pts, axis = 1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # multiply the rectangle by the original ratio
            rect *= resizeRatio
            # now that we have our rectangle of points, let's compute
            # the width of our new image
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # ...and now for the height of our new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            # take the maximum of the width and height values to reach
            # our final dimensions
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            # construct our destination points which will be used to
            # map the screen to a top-down, "birds eye" view
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
            # calculate the perspective transform matrix and warp
            # the perspective to grab the screen
            M = cv2.getPerspectiveTransform(rect, dst)
        
            warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

            self.bestWarpFound = warp
        else:
            warp = self.bestWarpFound
        # convert the warped image to grayscale and then adjust
        # the intensity of the pixels to have minimum and maximum
        # values of 0 and 255, respectively
        warp = exposure.rescale_intensity(warp, out_range = (0, 1))

        return warp