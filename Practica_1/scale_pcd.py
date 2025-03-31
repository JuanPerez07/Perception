"""
This script loads a point cloud, applies a translation along the X and Y axes, 
scales it down by a factor of 0.5, and saves the modified point cloud.
"""

import open3d as o3d
import numpy as np

# Directory and filename of the input point cloud
POINTCLOUD_DIR = "pointcloud/"
PCD_FILE = "carrapax_ajustado.pcd"  # Input file

# Load the point cloud
pcd = o3d.io.read_point_cloud(POINTCLOUD_DIR + PCD_FILE)

# Convert point cloud data into a NumPy array
points = np.asarray(pcd.points)

# Apply a translation along the X and Y axes
y_offset = 400  # Move the point cloud 400 units along the Y-axis
x_offset = -150  # Move the point cloud -150 units along the X-axis
points[:, 0] += x_offset  # Update X coordinates
points[:, 1] += y_offset  # Update Y coordinates

# Scale the point cloud by a factor of 0.5
scale_factor = 0.5  # Scaling factor
points *= scale_factor  # Apply scaling to all points

# Update the modified points back into the point cloud object
pcd.points = o3d.utility.Vector3dVector(points)

# Save the modified point cloud and visualize it
o3d.io.write_point_cloud(POINTCLOUD_DIR + "carrapax_movido.pcd", pcd)
o3d.visualization.draw_geometries([pcd])
