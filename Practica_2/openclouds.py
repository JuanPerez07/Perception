import open3d as o3d
import cv2 as cv
import numpy as np

POINTCLOUD_DIR= "clouds/objects/"
CLOUD_NAME="s0_piggybank_corr(1).pcd"
OBJ_NAME=POINTCLOUD_DIR+CLOUD_NAME
pcd = o3d.io.read_point_cloud(OBJ_NAME)

o3d.visualization.draw_geometries([pcd])