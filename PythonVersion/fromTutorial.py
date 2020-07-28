#Source for findBoard: https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
#Source for warpPerspective: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

# import the necessary packages
from skimage import exposure
import numpy as np
import imutils
import cv2

Precision = 0.015
NumCountours = 10
Canny1 = 30
Canny2 = 200

def findBoard(image):
    global Precision, NumCountours, Canny1, Canny2
    # compute the ratio of the old height
    # to the new height, clone it, and resize it
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17) # TODO is this needed?
    edged = cv2.Canny(gray, Canny1, Canny2)

    # FIND THE SCREEN
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:NumCountours]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, Precision * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    return screenCnt, ratio
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    # cv2.imshow("Game Boy Screen", image)
    # cv2.waitKey(0)

def warpPerspective(orig, screenCnt, resizeRatio):
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
    # convert the warped image to grayscale and then adjust
    # the intensity of the pixels to have minimum and maximum
    # values of 0 and 255, respectively
    warp = exposure.rescale_intensity(warp, out_range = (0, 1))

    return warp

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # get data from camera
    ret, image = capture.read()

    # process data
    try:
        contours, resizeRatio = findBoard(image)
        result = warpPerspective(image, contours, resizeRatio)

        # show result
        cv2.imshow('Whiteboard', result)
    except:
        print("some error occured")
        
    # print diagnostics
    diagnostics = '{},{},{},{}'.format(Precision, NumCountours, Canny1, Canny2)
    print(diagnostics)

    # the overly elaborate controls
    key = cv2.waitKey(1)
    if key == 27: # Escape
        break
    elif key == ord('w'): 
        Precision += 0.001
    elif key == ord('s'): 
        Precision -= 0.001
    elif key == ord('d'): 
        NumCountours += 1
    elif key == ord('a'): 
        NumCountours -= 1
   
    elif key == ord('w'):
        Canny1 += 1
    elif key == ord('S'):
        Canny1 -= 1
    elif key == ord('D'):
        Canny2 += 1
    elif key == ord('A'):
        Canny2 -= 1

# release resources
cv2.destroyAllWindows()