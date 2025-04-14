import open3d as o3d # type: ignore
import numpy as np
#from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#DIR SCENES
POINTCLOUD_DIR = "clouds/scenes/"
SCENE_NAME = "snap_0point.pcd"
ORIGINAL_CLOUD = POINTCLOUD_DIR + SCENE_NAME
OUTPUT_DIR = "clouds/scenes/"
#DIR OBJETOS
OBJ_DIR= "clouds/objects/"
MUG_NAME= "s0_mug_corr.pcd"
PIGGY_NAME= "s0_piggybank_corr.pcd"
PLANT_NAME= "s0_plant_corr.pcd"
PLC_NAME="s0_plc_corr.pcd"
#OBJ PCDS NAMES DIR
MUG=OBJ_DIR+MUG_NAME
PIGGY=OBJ_DIR+PIGGY_NAME
PLANT=OBJ_DIR+PLANT_NAME
PLC=OBJ_DIR+PLC_NAME



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
def deteccion_keypoints_iss():
    piggy_pcd=o3d.io.read_point_cloud(PIGGY)
    piggy_pcd=piggy_pcd.voxel_down_sample(voxel_size=0.005)#Bien porque si es menos hay pocos kypoints
    keypoints=o3d.geometry.keypoint.compute_iss_keypoints(piggy_pcd)

    #keypoints.paint_uniform_color([0,0,1]) 
    keypoints=keypoints_to_spheres(keypoints)#Mejorar visualmente los Kypoins
    piggy_pcd.paint_uniform_color([1,0,0]) #ver mejor el objeto con los key
    o3d.visualization.draw_geometries([keypoints,piggy_pcd])

#Ver los keypoints mejor en esferas
def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres


if __name__ == '__main__':
    
    # Ver nube de puntos original
    pcd = o3d.io.read_point_cloud(ORIGINAL_CLOUD)
    #o3d.visualization.draw_geometries([pcd],'Nube de puntos original')
    #Eliminacion de planos mediante RANSAC
    pcd= remove_planes_using_ransac(pcd)

    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    o3d.visualization.draw_geometries([pcd], 'Nube sin planos y con voxel')
    deteccion_keypoints_iss()