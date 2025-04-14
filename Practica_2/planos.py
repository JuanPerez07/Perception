import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


POINTCLOUD_DIR = "clouds/scenes/"
CLOUD_NAME = "snap_0point.pcd"
OBJ_NAME = POINTCLOUD_DIR + CLOUD_NAME
OUTPUT_DIR = "clouds/scenes/"



def remove_planes_using_ransac(pcd):
    plane_model, inliers= pcd.segment_plane(distance_threshold=0.03,ransac_n=3,num_iterations=1000)
    [a,b,c,d]= plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    inlier_cloud=pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],'Nube con plano')
    


if __name__ == '__main__':
    
    # Ver nube de puntos original
    
    pcd = o3d.io.read_point_cloud(OBJ_NAME)
    #o3d.visualization.draw_geometries([pcd],'Nube de puntos original')

    remove_planes_using_ransac(pcd)

    pcd = pcd.voxel_down_sample(voxel_size=0.01)

   