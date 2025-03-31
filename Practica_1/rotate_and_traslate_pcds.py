"""
This script was used to rotate and translate the point cloud (PCD) to align it with the real chessboard pattern.
"""

import cv2
import numpy as np
import glob
import os
import open3d as o3d  # Library to handle point cloud data (.pcd files)

# Directory and filenames for point cloud files
POINTCLOUD_DIR = "pointcloud/"
PCD_FILE = "carrapax.pcd"  # Input PCD file
OUTPUT_FILE = "carrapax_ajustado.pcd"  # Output PCD file after transformation

# Create a coordinate axis for visualization reference
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

# Load the point cloud from file
pcd = o3d.io.read_point_cloud(POINTCLOUD_DIR + PCD_FILE)

# Scale the point cloud by a factor of 30, keeping its center fixed
pcd.scale(30, center=pcd.get_center())

# Get the centroid of the point cloud
center = pcd.get_center()

# Translate the point cloud so that its centroid moves to the origin
pcd.translate(-center)

# Convert the point cloud to a NumPy array
points = np.asarray(pcd.points)

# Find the minimum Z value (formerly Y) to determine the lowest point of the cloud
x_min = np.min(points[:, 2])  

# Define a 180-degree rotation matrix around the X, Y, and Z axes
theta = np.radians(180)  # Convert degrees to radians

rotation_matrix_X = np.array([
    [1, 0, 0],
    [0, np.cos(theta),  np.sin(theta)],
    [0, -np.sin(theta), np.cos(theta)], 
])

rotation_matrix_Y = np.array([
    [np.cos(theta),  0, np.sin(theta)],
    [0,              1, 0             ],
    [-np.sin(theta), 0, np.cos(theta)]
])

rotation_matrix_Z = np.array([
    [np.cos(theta),  np.sin(theta), 0],
    [-np.sin(theta), np.cos(theta), 0], 
    [0, 0, 1]
])

# Apply rotation around the X-axis
pcd.rotate(rotation_matrix_X, center=(0, 0, 0))

# Compute the translation offset to lift the cloud upwards
x_offset = -x_min  

# Translate the point cloud upwards along the Z-axis
pcd.translate((0, 0, -x_offset))  

# Paint the entire point cloud in blue for visualization
pcd.paint_uniform_color([0, 0, 1])

# Save the transformed point cloud to a new file
o3d.io.write_point_cloud(POINTCLOUD_DIR + OUTPUT_FILE, pcd)

# Visualize the adjusted point cloud along with the coordinate axis
o3d.visualization.draw_geometries([pcd, axis])
