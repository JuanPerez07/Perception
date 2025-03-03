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
PCD_FILE = POINTCLOUD_DIR + "cortacuellos.pcd"
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


def load_point_cloud():
    """Loads the first available .pcd point cloud file from the POINTCLOUD_DIR."""
    pcd_files = glob.glob(POINTCLOUD_DIR + "*.pcd")
    
    if not pcd_files:
        print("⚠ No .pcd files found in 'pointcloud/' directory!")
        return None
    
    # Read the first available .pcd file
    pcd = o3d.io.read_point_cloud(PCD_FILE)
    points = np.asarray(pcd.points, dtype=np.float32)  # Convert to numpy array
    
    print(f"✅ Loaded {points.shape[0]} points from {PCD_FILE}")
    return points


def project_point_cloud(frame, camera_matrix, dist_coeffs, point_cloud):
    """Projects a 3D point cloud onto the detected chessboard dynamically."""
    found, corners, objp = detect_chessboard(frame)

    if found:
        # Compute rotation and translation vectors dynamically
        ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

        if ret:
            # Project the 3D point cloud to 2D
            imgpts, _ = cv2.projectPoints(point_cloud, rvec, tvec, camera_matrix, dist_coeffs)

            # Draw projected points on the frame
            for pt in imgpts:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    return frame


def overlay_pointcloud_live():
    """Captures video feed and overlays the projected point cloud in real-time."""
    print("📷 Starting video capture with 3D point cloud overlay...")
    camera_matrix, dist_coeffs = load_camera_parameters()
    point_cloud = load_point_cloud()

    if point_cloud is None:
        print("❌ Exiting: No point cloud loaded.")
        return

    cap = cv2.VideoCapture(1)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Project the point cloud onto the frame
        frame = project_point_cloud(frame, camera_matrix, dist_coeffs, point_cloud)

        cv2.imshow("Augmented Reality - Point Cloud", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_pointcloud_live()
