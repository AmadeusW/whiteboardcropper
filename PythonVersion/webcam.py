import cv2

def processImage(image):
    return image

# get the image from the video camera
capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    result = processImage(frame)
    cv2.imshow('Whiteboard Viewer', result)
    if cv2.waitKey(1) == 27: # Escape
        break

# release resources
capture.release()
cv2.destroyAllWindows()
