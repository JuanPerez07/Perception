import cv2
import numpy as np
import glob
import os
import open3d as o3d  # To read .pcd files


POINTCLOUD_DIR = "pointcloud/"
PCD_FILE = "gormiti.pcd"
OUTPUT_FILE = "cortacuellos.pcd"

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

#pcd_files = glob.glob(POINTCLOUD_DIR + "*.pcd") # ?
pcd = o3d.io.read_point_cloud(POINTCLOUD_DIR + PCD_FILE)
pcd.scale(30, center=pcd.get_center())
center = pcd.get_center()  # Obtener centroide
pcd.translate(-center)  # Mover nube al origen


points = np.asarray(pcd.points)
x_min = np.min(points[:, 2])  # Encuentra el valor mínimo en Y


theta = np.radians(180)  # Convertir grados a radianes
rotation_matrix_X = np.array([
    [1,0,0],
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
    [0,0,1]
])
pcd.rotate(rotation_matrix_X, center=(0, 0, 0))

x_offset = -x_min  # Mover la nube hacia arriba
pcd.translate((0,0,-x_offset))  # Trasladar solo en Y
pcd.paint_uniform_color([1, 0, 0])

o3d.io.write_point_cloud(POINTCLOUD_DIR + OUTPUT_FILE, pcd)
o3d.visualization.draw_geometries([pcd,axis])
