import cv2
import numpy as np


def calibrate(image):
    rect = cv2.selectROI(image, True, False)
    return rect

# TODO add some processing
# beware super complicated processing going on below
def processImage(image, cropRect):
    return image[int(cropRect[1]):int(cropRect[1]+cropRect[3]), int(cropRect[0]):int(cropRect[0]+cropRect[2])]

def OpenCamera():
    # get the image from the video camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    # get the rect to crop. This helps us know where the board is
    ret, intialImg = capture.read()
    cropRect = calibrate(intialImg)

    while True: # <-- yuck python with your capitalize bools
        ret, frame = capture.read()

        result = processImage(frame, cropRect)

        cv2.imshow("Whiteboard Viewer", result)

        #exit the app if escape key is pressed
        if cv2.waitKey(1) == 27:
            break

    # release resources
    capture.release()
    cv2.destroyAllWindows()

def GoFromImage():
    #open the image and convert it to gray scale image
    main_image = cv2.imread('samples/3.jpg')

    result = processImage(main_image)

    cv2.imshow("Whiteboard Viewer", result)

    # delay exit
    input("Press Enter to continue...")

    cv2.destroyAllWindows()

OpenCamera()
# GoFromImage()

#############################################################################################
#TODO
# Find whiteboard and crop by it
# Raise contrast 
# Flip preview
# Find more todos

#Sources
# https://stackoverflow.com/questions/41995916/opencv-straighten-an-image-with-python