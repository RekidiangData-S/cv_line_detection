import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
        [(0, height), (800, height), (380, 290)]
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment


cap = cv.VideoCapture("input.mp4")
while (cap.isOpened()):
    ret, frame = cap.read()

    cv.imshow('frame', frame)

    canny = do_canny(frame)
    cv.imshow('canny', canny)

    segment = do_segment(frame)
    cv.imshow('segment', segment)

    canny_segment = do_segment(canny)
    cv.imshow('canny_segment', canny_segment)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break
"""
    plt.imshow(frame)
    plt.show()
    plt.imshow(canny)
    plt.show()
    plt.imshow(segment)
    plt.show()
    plt.imshow(segment_canny)
    plt.show()
"""


cap.release()
cv.destroyAllWindows()
