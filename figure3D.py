import cv2
import numpy as np
import glob
import os
import open3d as o3d  # To read .pcd files

# Directories
PARAM_DIR = "params/"
POINTCLOUD_DIR = "pointcloud/"
CHESSBOARD_SIZE = (6, 9)  # Internal corners per row and column

def load_camera_parameters():
    """Loads intrinsic and extrinsic parameters from PARAM_DIR."""
    camera_matrix = np.load(PARAM_DIR + "camera_matrix.npy")
    dist_coeffs = np.load(PARAM_DIR + "dist_coeffs.npy")
    extrinsics = np.load(PARAM_DIR + "extrinsics.npy", allow_pickle=True)
    return camera_matrix, dist_coeffs, extrinsics


def detect_chessboard(img):
    """Detects chessboard corners in an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if found:
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return True, refined_corners
    return False, None


def load_point_cloud():
    """Loads the first available .pcd point cloud file from the POINTCLOUD_DIR."""
    pcd_files = glob.glob(POINTCLOUD_DIR + "*.pcd")
    
    if not pcd_files:
        print("⚠ No .pcd files found in 'pointcloud/' directory!")
        return None
    
    # Read the first available .pcd file
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    points = np.asarray(pcd.points, dtype=np.float32)  # Convert to numpy array
    
    print(f" Loaded {points.shape[0]} points from {pcd_files[0]}")
    return points


def project_point_cloud(frame, camera_matrix, dist_coeffs, extrinsics, point_cloud):
    """Projects a 3D point cloud onto the detected chessboard."""
    found, corners = detect_chessboard(frame)

    if found:
        R = extrinsics[0][:, :3]  # Rotation matrix
        t = extrinsics[0][:, 3]   # Translation vector
        rvec, _ = cv2.Rodrigues(R)

        # Project 3D points to 2D
        imgpts, _ = cv2.projectPoints(point_cloud, rvec, t, camera_matrix, dist_coeffs)

        # Draw projected points on the frame
        for pt in imgpts:
            x, y = int(pt[0][0]), int(pt[0][1])
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    return frame


def overlay_pointcloud_live():
    """Captures video feed and overlays the projected point cloud in real-time."""
    print(" Starting video capture with 3D point cloud overlay...")
    camera_matrix, dist_coeffs, extrinsics = load_camera_parameters()
    point_cloud = load_point_cloud()

    if point_cloud is None:
        print(" Exiting: No point cloud loaded.")
        return

    cap = cv2.VideoCapture(0)  # Open webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Project the point cloud onto the frame
        frame = project_point_cloud(frame, camera_matrix, dist_coeffs, extrinsics, point_cloud)

        cv2.imshow("Augmented Reality - Point Cloud", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    overlay_pointcloud_live()
