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
OBJETOS=[MUG,PIGGY,PLANT,PLC]



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
def deteccion_keypoints_iss(pcd_scene):

    voxel_size=0.005

    """Detección de keypoints de la escena
    primero estimamos las normales"""
    pcd_scene=pcd_scene.voxel_down_sample(voxel_size=voxel_size)
    pcd_scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,max_nn=30))
   
    #Calculo de keypoints con ISS
    key_scene=o3d.geometry.keypoint.compute_iss_keypoints(pcd_scene,
                                            salient_radius=0.005,
                                            non_max_radius=0.005,
                                            gamma_21=0.5,
                                            gamma_32=0.5)#Obtencion de los descriptores mediante ISS
    #key_scene=keypoints_to_spheres(key_scene) #relmente no hace falta pasarlos
    key_scene.paint_uniform_color([1,0,1])
    pcd_scene.paint_uniform_color([0,0.5,0.5])#pintar la nube escena para diferenciar
    print("Visualizando keypoints de la escena")
    o3d.visualization.draw_geometries([pcd_scene,key_scene])

    """Obtener descriptores de la escena mediante FPFH"""
    key_scene_fpfh=descriptores_fpfh(key_scene,pcd_scene,voxel_size)
    print("FPFH",key_scene_fpfh)


    """Obtener keypoints de los objetos"""
   
    piggy_pcd=o3d.io.read_point_cloud(PIGGY)
    piggy_pcd=piggy_pcd.voxel_down_sample(voxel_size=voxel_size)#Bien porque si es menos hay pocos kypoints
    piggy_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01,max_nn=30))

    key_piggy=o3d.geometry.keypoint.compute_iss_keypoints(piggy_pcd,
                                                          salient_radius=0.005,
                                                          non_max_radius=0.005,
                                                          gamma_21=0.5,
                                                          gamma_32=0.5)
    piggy_fpfh=descriptores_fpfh(key_piggy,piggy_pcd,voxel_size)
    #keypoints.paint_uniform_color([0,0,1]) 
    key_piggy=keypoints_to_spheres(key_piggy)#Mejorar visualmente los Kypoins
    piggy_pcd.paint_uniform_color([1,0,0]) #ver mejor el objeto con los key
    o3d.visualization.draw_geometries([key_piggy,piggy_pcd])

#Ver los keypoints mejor en esferas 
def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres
def descriptores_fpfh(key,pcd,voxel_size):

    radius_normal=voxel_size*2
    
    #key.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal,max_nn=30))
    radius_feature=voxel_size*5
    fpfh=o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    #Filtrar para solo los keypoints
    keypoint_indices = np.asarray(key.points).astype(int).flatten()
    keypoint_fpfh = fpfh.data[:, keypoint_indices]

    return keypoint_fpfh

if __name__ == '__main__':
    
    # Ver nube de puntos original
    pcd = o3d.io.read_point_cloud(ORIGINAL_CLOUD)
    #o3d.visualization.draw_geometries([pcd],'Nube de puntos original')
    #Eliminacion de planos mediante RANSAC
    pcd_scene= remove_planes_using_ransac(pcd)

    pcd_scene = pcd_scene.voxel_down_sample(voxel_size=0.005)
    o3d.visualization.draw_geometries([pcd_scene], 'Nube sin planos y con voxel')
    deteccion_keypoints_iss(pcd_scene)