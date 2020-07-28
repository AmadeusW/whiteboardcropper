import cv2
import numpy as np

# https://stackoverflow.com/a/45579542/879243
def processImage(image):
    edges = basicEdgeDetection(image)
    #return lineDetection(edges, image)
    return contourDetection(edges, image)

def lineDetection(edges, image):
    # Now that we've detected edges, find lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    points = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges

def contourDetection(edges, image):
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_image = np.copy(image) * 0  # creating a blank to draw lines on
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(line_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return cv2.addWeighted(image, 0.8, line_image, 1, 0)

def basicEdgeDetection(image):
    # Basic edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0.5) # TODO: Adjust the parameters; GaussianBlur(gray,(kernel_size, kernel_size),0)
    edges = cv2.Canny(blur, 0, 50, 3) # TODO: Adjust the thresholds; Canny(blur_gray, low_threshold, high_threshold)
    return edges

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
