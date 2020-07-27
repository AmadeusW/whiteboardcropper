import cv2
import numpy as np

#open the image and convert it to gray scale image
# main_image = cv2.imread('samples/1.jpg')
# gray_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)


def processImage(image):
    #---- 4 corner points of the bounding box
    pts_src = np.array([[17.0,0.0], [77.0,5.0], [0.0, 552.0],[53.0, 552.0]])

    #---- 4 corner points of the black image you want to impose it on
    pts_dst = np.array([[0.0,0.0],[77.0, 0.0],[ 0.0,552.0],[77.0, 552.0]])

    #---- forming the black image of specific size
    image_dst = np.zeros((552, 77, 3), np.uint8)

    #---- Framing the homography matrix
    h, status = cv2.findHomography(pts_src, pts_dst)

    #---- transforming the image bound in the rectangle to straighten
    im_out = cv2.warpPerspective(image, h, (image_dst.shape[1],image_dst.shape[0]))
    
    return im_out

# get the image from the video camera
capture = cv2.VideoCapture(0)

while True: # <-- yuck python with your capitalize bools
    ret, frame = capture.read()

    result = processImage(frame)

    cv2.imshow('Whiteboard Viewer', result)

    #exit the app if escape key is pressed
    if cv2.waitKey(1) == 27:
        break

# release resources
capture.release()
cv2.destroyAllWindows()

#############################################################################################
#TODO
# Find whiteboard and crop by it
# Raise contrast 
# Flip preview
# Find more todos

#Sources
# https://stackoverflow.com/questions/41995916/opencv-straighten-an-image-with-python