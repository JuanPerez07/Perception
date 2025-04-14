import open3d as o3d # type: ignore
import numpy as np
#from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


POINTCLOUD_DIR = "clouds/scenes/"
CLOUD_NAME = "snap_0point.pcd"
OBJ_NAME = POINTCLOUD_DIR + CLOUD_NAME
OUTPUT_DIR = "clouds/scenes/"
PLANOS= 3


def remove_planes_using_ransac(pcd):
    """ Eliminacion de nubes de puntos mediante ransac"""
    for _ in range (PLANOS):

        _, inliers= pcd.segment_plane(distance_threshold=0.03,ransac_n=5,num_iterations=1000)
        
        
        
        inlier_cloud=pcd.select_by_index(inliers)
        #inlier_cloud.paint_uniform_color([1.0, 0, 0]) #Pintar los planos para verlos
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([outlier_cloud],'Nube sin el plano')
        #o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud],'Nube con el plano')
        pcd= outlier_cloud
    return pcd
if __name__ == '__main__':
    
    # Ver nube de puntos original
    pcd = o3d.io.read_point_cloud(OBJ_NAME)
    #o3d.visualization.draw_geometries([pcd],'Nube de puntos original')
    #Eliminacion de planos mediante RANSAC
    pcd= remove_planes_using_ransac(pcd)

    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    o3d.visualization.draw_geometries([pcd], 'Nube sin planos y con voxel')
   