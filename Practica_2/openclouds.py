import open3d as o3d
import cv2 as cv
import numpy as np

POINTCLOUD_DIR= "clouds/scenes/"
CLOUD_NAME="snap_0point.pcd"
OBJ_NAME=POINTCLOUD_DIR+CLOUD_NAME
pcd = o3d.io.read_point_cloud(OBJ_NAME)
OUTPUT_DIR="clouds/scenes/"

# function to remove the planes using RANSAC
def remove_planes(pcd, threshold=0.02, iterations=1000, max_planes=3):
    remaining_pcd = pcd
    for _ in range(max_planes):
        plane_model, inliers = remaining_pcd.segment_plane(distance_threshold=threshold,
                                                           ransac_n=3,
                                                           num_iterations=iterations)
        # Extract the plane normal
        normal = np.abs(plane_model[:3])  # Keep absolute to handle flipped normals

        # If the normal aligns with common planes (horizontal or vertical), remove the plane
        if normal[1] > 0.9 or normal[0] > 0.9 or normal[2] > 0.9:  # Ceiling, floor, walls
            print(f"Removing plane with normal: {normal}")
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        else:
            break  # Stop if no more large planes are found

    return remaining_pcd

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(OUTPUT_DIR + "original.ply", pcd)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Normalize normal vectors
normals = np.asarray(pcd.normals)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)

# Apply plane removal
filtered_pcd = remove_planes(pcd)
o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
o3d.io.write_point_cloud(OUTPUT_DIR + "resultados.ply", filtered_pcd)
