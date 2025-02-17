#!/usr/bin/env python3
import os
import cv2 as cv
from pathlib import Path
import numpy as np
import glob
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

SQUARE_SIZE=25

DATA_FILES = [file for file in Path(DATA_DIR).iterdir() if file.suffix == ".png"]
#DATA_FILES = glob.glob(DATA_FILES)


RESIZE_FORMAT = (1280,720)
CHESSBOARD_SIZE = (8,8)

# Listas para almacenar puntos 3D (mundo real) y puntos 2D (imagen)
objpoints = []  # Puntos 3D
imgpoints = []  # Puntos 2D detectados en las imágenes

# Puntos 3D en el espacio real del ajedrez (coordenadas en milímetros)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


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

def find_corners(img):
    # Buscar esquinas del patrón de ajedrez
    ret, corners = cv.findChessboardCorners(img, CHESSBOARD_SIZE, None)

    if ret:  # Si se encontraron las esquinas
        print("Esquinas found")
        objpoints.append(objp)
        refined_corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

        # Dibujar las esquinas detectadas
        cv.drawChessboardCorners(img, CHESSBOARD_SIZE, refined_corners, ret)
        cv.imshow('image', img)
        cv.waitKey(0)
    else:
        print("No entra")

if __name__ == '__main__':
    for f in DATA_FILES:
        img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)  # Read as grayscale
        if img is None:
            print(f"Warning: Could not read image {f}")
            continue

        #edges = compute_edge_detection(img)  # Compute edges

        find_corners(img)        
        
        """
        # Save the result in the 'edges' directory
        filename = os.path.join(EDGE_DIR, 'edge_' + f.name)
        
        dst = find_corners(img)

        cv.imwrite(filename, dst)

        print(f"Processed and saved: {filename}")
        """