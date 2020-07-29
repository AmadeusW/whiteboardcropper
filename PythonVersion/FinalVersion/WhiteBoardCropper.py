#Source for findBoard: https://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
#Source for warpPerspective: https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/

# import the necessary packages
from skimage import exposure
import numpy as np
import imutils
import cv2
import math 

#import our classes
from ImageEnhancer import ImageEnhancer
from ImageWarper import ImageWarper
from WhiteBoardFinder import WhiteBoardFinder

# adjustments
Precision = 0.12
Canny1 = 30
Canny2 = 265
ShowCalibrationWindow = False


print(
" __          ___     _ _       _                         _    _____                                  \n" +
" \ \        / / |   (_) |     | |                       | |  / ____|                                 \n" +
"  \ \  /\  / /| |__  _| |_ ___| |__   ___   __ _ _ __ __| | | |     _ __ ___  _ __  _ __   ___ _ __  \n" +
"   \ \/  \/ / | '_ \| | __/ _ \ '_ \ / _ \ / _` | '__/ _` | | |    | '__/ _ \| '_ \| '_ \ / _ \ '__| \n" +
"    \  /\  /  | | | | | ||  __/ |_) | (_) | (_| | | | (_| | | |____| | | (_) | |_) | |_) |  __/ |    \n" +
"     \/  \/   |_| |_|_|\__\___|_.__/ \___/ \__,_|_|  \__,_|  \_____|_|  \___/| .__/| .__/ \___|_|    \n" +
"                                                                             | |   | |               \n" +
"                                                                             |_|   |_|               \n" +
"Controls:                                                                                            \n" +
"q = exit    r = reset    c = toggle calibration window                                               \n"
)

# initialize our classes
whiteBoardFinder = WhiteBoardFinder(Precision, Canny1, Canny2)
imageWarper = ImageWarper(whiteBoardFinder)
imageEnhancer = ImageEnhancer()

# start the camera
capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
if not capture.isOpened():
    print("Cannot open camera")
    exit()

# start render loop
while True:
    # get data from camera
    ret, image = capture.read()

    # find the board
    contours, resizeRatio, annotatedImg, edges, area = whiteBoardFinder.findBoard(image)

    if not ShowCalibrationWindow and contours is not None and len(contours) > 0:
        # process the image
        cropped = imageWarper.warpPerspective(image, contours, resizeRatio, area)
        annotatedImg = imageEnhancer.enhance(cropped)

    cv2.imshow('Whiteboard Cropper', imutils.resize(annotatedImg, height = 300))

    # the overly elaborate controls
    key = cv2.waitKey(1)
    if key == ord('q'): # exit 
        break

    elif key == ord('r'): # reset
        imageWarper.resetCache()
        whiteBoardFinder.resetCache()

    elif key == ord('c'): #calibration window
        ShowCalibrationWindow = not ShowCalibrationWindow

# release resources
capture.release()
cv2.destroyAllWindows()