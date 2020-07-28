import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1):
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  
  # todo make these global constants
  main_image = cv2.imread('samples/straight_padding.jpg')
  im2Gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)

  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)

  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  imMatches = cv2.drawMatches(im1, keypoints1, im2Gray, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
  
  # Use homography
  height, width = im2Gray.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))

  return im1Reg, h

def processImage(image):
    image, h = alignImages(image)

    return image

def OpenCamera():
    # get the image from the video camera
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()

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

def GoFromImage():
    #open the image and convert it to gray scale image
    main_image = cv2.imread('samples/3.jpg')

    result = processImage(main_image)

    cv2.imshow('Whiteboard Viewer', result)

    # delay exit
    input("Press Enter to continue...")

    cv2.destroyAllWindows()

#OpenCamera()
GoFromImage()

#############################################################################################
#TODO
# Find whiteboard and crop by it
# Raise contrast 
# Flip preview
# Find more todos

#Sources
# https://stackoverflow.com/questions/41995916/opencv-straighten-an-image-with-python