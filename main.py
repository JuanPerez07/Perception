import cv2
import numpy as np
import glob
import os

# Dataset directory and image format
DATA_DIR = "dataset"
IMG_FORMAT = "/*.png"
# Camera parameter directory
PARAM_DIR = "params/"

# Chessboard pattern size (internal corners per row and column)
CHESSBOARD_SIZE = (6, 9)  # Adjust according to your pattern
SQUARE_SIZE = 29  # Square size in mm (adjust accordingly)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points in the real world (in mm)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Storage for 3D and 2D points
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

def calculate_params(images):
    """ Calculates and saves the intrinsic and extrinsic parameters of the camera. """
    img = cv2.imread(images[0])
    height, width = img.shape[:2]
    image_shape = (width, height)  # OpenCV expects (width, height)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(refined_corners)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    extrinsics = []
    for i in range(min(len(rvecs), len(tvecs))):
        R, _ = cv2.Rodrigues(rvecs[i])
        T = np.hstack((R, tvecs[i]))  # [R | t] matrix
        extrinsics.append(T)

    np.save(PARAM_DIR + "camera_matrix.npy", camera_matrix)
    np.save(PARAM_DIR + "dist_coeffs.npy", dist_coeffs)
    np.save(PARAM_DIR + "extrinsics.npy", extrinsics)

    return objpoints, imgpoints

def compute_error(objpoints, imgpoints):
    """ Computes the reprojection error using stored calibration parameters. """
    camera_matrix = np.load(PARAM_DIR + "camera_matrix.npy")
    dist_coeffs = np.load(PARAM_DIR + "dist_coeffs.npy")
    extrinsics = np.load(PARAM_DIR + "extrinsics.npy", allow_pickle=True)

    total_error = 0
    for i in range(len(objpoints)):
        R = extrinsics[i][:, :3]
        t = extrinsics[i][:, 3]
        rvec, _ = cv2.Rodrigues(R)

        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, t, camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    return mean_error

if __name__ == '__main__':
    print("Calculating camera parameters...")
    
    images = glob.glob(DATA_DIR + IMG_FORMAT)
    
    if not images:
        print(" No images found in dataset directory!")
        exit()

    img_points, obj_points = calculate_params(images)
    error = compute_error(img_points, obj_points)
    print(f" Mean Reprojection Error: {error}")

