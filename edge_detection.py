#!/usr/bin/env python3
import os
import cv2 as cv
from pathlib import Path

"""
The objective is to compute edges across different perspectives of the same pattern
    (chessboard) to gain intrinsic and extrinsic parameters of the camera used.

Intrinsic stays the same.
Extrinsic changes through photos.
"""

# Data directories and files
DATA_DIR = os.path.join(os.getcwd(), "data")
EDGE_DIR = os.path.join(os.getcwd(), "edges")
os.makedirs(EDGE_DIR, exist_ok=True)  # Create 'edges' directory if it doesn't exist

DATA_FILES = [file for file in Path(DATA_DIR).iterdir() if file.suffix == ".png"]

RESIZE_FORMAT = (1280,720)
CHESSBOARD_SIZE = (8,8)
# Returns the image scale to a size of 
def resize(img):
    return cv.resize(img, RESIZE_FORMAT)
# Returns a grayscale image blurred with a kernel of size 5 and 0 deviation
def gaussian_blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)

# Compute the edge using Canny detector
def compute_edge_detection(img):
    img = resize(img)
    blurred = gaussian_blur(img)
    edges = cv.Canny(blurred, 100, 200)  # Threshold values can be tuned
    return edges

# Compute the corner detector using Harris
def compute_corners_detector(img):
    ret, corners = cv.findChessboardCorners(img, CHESSBOARD_SIZE, None)    

    if ret: # corners detected
        cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        return img
    
    return img 



if __name__ == '__main__':
    """
    for f in DATA_FILES:
        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            print(f"Warning: Could not read image {f}")
            continue

        edges = compute_edge_detection(img)  # Compute edges
        
        dst = compute_corners_detector(edges)
       
        # Save the result in the 'edges' directory
        filename = os.path.join(EDGE_DIR, 'edge_' + f.name)
        
        cv.imwrite(filename, dst)

        print(f"Processed and saved: {filename}")
    """
    for f in DATA_FILES:
        src = resize(cv.imread(str(f), cv.IMREAD_GRAYSCALE))
        blur_src = gaussian_blur(src)
        dst = compute_corners_detector(blur_src)
        cv.imshow("Corners", dst)
        cv.waitKey(0)
