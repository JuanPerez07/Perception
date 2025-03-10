import cv2
import numpy as np
import glob
import os
import open3d as o3d  # To read .pcd files

# Directories
PARAM_DIR = "params/"
POINTCLOUD_DIR = "pointcloud/"
CHESSBOARD_SIZE = (6, 9)  # Internal corners per row and column
SQUARE_SIZE = 29  # Square size in mm (match calibration)

# PCD file names
CORTACUELLOS_FILE = POINTCLOUD_DIR + "cortacuellos_movido.pcd"
CARRAPAX_FILE = POINTCLOUD_DIR + "carrapax_movido.pcd"

def load_camera_parameters():
    """Loads intrinsic parameters from PARAM_DIR."""
    camera_matrix = np.load(PARAM_DIR + "camera_matrix.npy")
    dist_coeffs = np.load(PARAM_DIR + "dist_coeffs.npy")
    return camera_matrix, dist_coeffs


def detect_chessboard(img):
    """Detects chessboard corners and estimates extrinsics in each frame."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Define 3D object points relative to the chessboard
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

        return True, refined_corners, objp
    return False, None, None


def load_point_clouds():
    """Loads two different point clouds and positions them opposite to each other in the XY plane."""
    
    # Load cortacuellos (original position)
    cortacuellos_pcd = o3d.io.read_point_cloud(CORTACUELLOS_FILE)
    cortacuellos_points = np.asarray(cortacuellos_pcd.points, dtype=np.float32)

    # Compute bounding box to get the furthest point (XY plane only)
    min_bound = np.min(cortacuellos_points, axis=0)
    max_bound = np.max(cortacuellos_points, axis=0)
    center_xy = (min_bound[:2] + max_bound[:2]) / 2  # Only X and Y

    # Compute offset in XY plane
    offset_vector = np.zeros(3)
    offset_vector[:2] = 2 * (max_bound[:2] - center_xy)  # Apply translation only in X and Y

    # Load carrapax_ajustado
    carrapax_pcd = o3d.io.read_point_cloud(CARRAPAX_FILE)
    carrapax_points = np.asarray(carrapax_pcd.points, dtype=np.float32)

    # Apply XY offset to carrapax, Z remains unchanged
    carrapax_points[:, :2] += offset_vector[:2]

    print(f"✅ Loaded {cortacuellos_points.shape[0]} points from '{CORTACUELLOS_FILE}'")
    print(f"✅ Loaded {carrapax_points.shape[0]} points from '{CARRAPAX_FILE}', moved by {offset_vector[:2]} in XY plane")

    return cortacuellos_points, carrapax_points


def project_point_clouds(frame, camera_matrix, dist_coeffs, point_cloud_1, point_cloud_2):
    """Projects two 3D point clouds onto the detected chessboard dynamically."""
    found, corners, objp = detect_chessboard(frame)

    if found:
        # Compute rotation and translation vectors dynamically
        ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

        if ret:
            # Project cortacuellos
            imgpts1, _ = cv2.projectPoints(point_cloud_1, rvec, tvec, camera_matrix, dist_coeffs)
            for pt in imgpts1:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)  # Red color for cortacuellos

            # Project carrapax
            imgpts2, _ = cv2.projectPoints(point_cloud_2, rvec, tvec, camera_matrix, dist_coeffs)
            for pt in imgpts2:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue color for carrapax

    return frame


def overlay_pointclouds_live():
    """Captures video feed and overlays two projected point clouds in real-time."""
    print("📷 Starting video capture with 3D point cloud overlay...")
    camera_matrix, dist_coeffs = load_camera_parameters()
    point_cloud_1, point_cloud_2 = load_point_clouds()

    cap = cv2.VideoCapture(1)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Project both point clouds onto the frame
        frame = project_point_clouds(frame, camera_matrix, dist_coeffs, point_cloud_1, point_cloud_2)

        cv2.imshow("Augmented Reality - Dual Point Cloud", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_pointclouds_live()
