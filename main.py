import cv2
import numpy as np
import glob

# Dataset directory and image format
DATA_DIR = "dataset"
IMG_FORMAT = "/*.png"

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

# Load all images from the dataset folder
images = glob.glob(DATA_DIR + IMG_FORMAT)

# Get image size from the first image
img = cv2.imread(images[0])
height, width = img.shape[:2]
image_shape = (width, height)  # OpenCV expects (width, height)

# Detect chessboard corners in images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

# ** Camera Calibration **
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_shape, None, None
)

# ** Compute Extrinsic Matrices for Each Image **
extrinsics = []
for i in range(min(len(rvecs),len(tvecs))):
    print(f"Valor de i: {i}")
    R, _ = cv2.Rodrigues(rvecs[i])  # Convert rotation vector to matrix
    T = np.hstack((R, tvecs[i]))  # [R | t] matrix
    extrinsics.append(T)

# ** Save Intrinsic and Extrinsic Parameters **
np.save("camera_matrix.npy", camera_matrix)  # Intrinsic matrix
np.save("dist_coeffs.npy", dist_coeffs)  # Distortion coefficients
np.save("extrinsics.npy", extrinsics)  # Extrinsic matrices

# ** Compute Reprojection Error **
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

print(f"Mean Reprojection Error: {total_error / len(objpoints)}")

cv2.destroyAllWindows()
