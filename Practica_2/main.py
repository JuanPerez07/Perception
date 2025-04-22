import open3d as o3d  # type: ignore
import numpy as np
# from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# DIR SCENES
POINTCLOUD_DIR = "clouds/scenes/"
SCENE_NAME = "snap_0point.pcd"
ORIGINAL_CLOUD = POINTCLOUD_DIR + SCENE_NAME
OUTPUT_DIR = "clouds/scenes/"

# DIR OBJETOS
OBJ_DIR = "clouds/objects/"
MUG_NAME = "s0_mug_corr.pcd"
PIGGY_NAME = "s0_piggybank_corr.pcd"
PLANT_NAME = "s0_plant_corr.pcd"
PLC_NAME = "s0_plc_corr.pcd"

# OBJ PCDS NAMES DIR
MUG = OBJ_DIR + MUG_NAME
PIGGY = OBJ_DIR + PIGGY_NAME
PLANT = OBJ_DIR + PLANT_NAME
PLC = OBJ_DIR + PLC_NAME
OBJETOS = [MUG, PIGGY, PLANT, PLC]

PLANOS = 3
"""
Remove planes using RANSAC
"""
def remove_planes_using_ransac(pcd):
    for i in range(PLANOS):
        # detect a plane in the cloud
        _, inliers = pcd.segment_plane(
                distance_threshold=0.015,  # distancia máxima entre un punto y el plano para considerarlo parte de él
                ransac_n=3,               # número de puntos aleatorios usados para estimar un plano
                num_iterations=1000       # número de iteraciones para encontrar el mejor plano
        )
        # keep only the outliers
        pcd = pcd.select_by_index(inliers, invert=True)
        # save the pcd without the inliers (only outliers)
        o3d.io.write_point_cloud(f"{OUTPUT_DIR}step_ransac_{i}.ply", pcd)

    return pcd
"""
Downsample using voxelgrid
"""
def downsample_pcd(pcd, vx_size):
    return pcd.voxel_down_sample(voxel_size=vx_size)

"""
Detect the keypoints of an object and its scene using ISS
"""
def detect_keypoints_iss(pcd_scene, pcd_object):
    # Estimación de normales de la escena
    pcd_scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # Calculo keypoints escena por ISS
    key_scene = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_scene,
        salient_radius=0.005,
        non_max_radius=0.005,
        gamma_21=0.5,
        gamma_32=0.5,
    )
    # pintar los keypoints
    key_scene.paint_uniform_color([1, 0, 1])
    pcd_scene.paint_uniform_color([0, 0.5, 0.5])
    #print("Keypoints de la escena detectados")

    # Guardar escena y keypoints
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_downsampled.ply", pcd_scene)
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_keypoints.ply", key_scene)

    # OBJETO: PIGGYBANK
    piggy_pcd = pcd_object
    # Estimar normales
    piggy_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # Calculo de keypoints por ISS
    key_piggy = o3d.geometry.keypoint.compute_iss_keypoints(
        piggy_pcd,
        salient_radius=0.005,
        non_max_radius=0.005,
        gamma_21=0.5,
        gamma_32=0.5
    )
    o3d.io.write_point_cloud(f"{OBJ_DIR}piggy_kp_iss.ply", key_piggy)
    # reducir keypoints a esferas
    key_piggy = keypoints_to_spheres(key_piggy)
    piggy_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([key_piggy, piggy_pcd])
    o3d.io.write_triangle_mesh(f"{OBJ_DIR}piggy_keypoints_spheres.ply", key_piggy)
    
    # Return the scene and object keypoints
    return key_scene, key_piggy

"""
Simplify the keypoints into clustered spheres using a TriangleMesh 
"""
def keypoints_to_spheres(keypoints):
    spheres = o3d.geometry.TriangleMesh()
    for keypoint in keypoints.points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(keypoint)
        spheres += sphere
    spheres.paint_uniform_color([1.0, 0.75, 0.0])
    return spheres

"""
Compute the FPFH descriptor for a given pcd and the list of keypoints from ISS
"""
def descript_fpfh(key, pcd, voxel_size=0.005):
    #radius_normal = voxel_size * 2 
    radius_feature = voxel_size * 5 # neighbourhood size
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    # find the idx -> key must be a subset of pcd
    keypoint_indices = np.asarray(key.points).astype(int).flatten()
    keypoint_fpfh = fpfh.data[:, keypoint_indices]

    return keypoint_fpfh

if __name__ == '__main__':
    # load both scene and objects pcds
    piggy_pcd = o3d.io.read_point_cloud(PIGGY) # object
    scene_pcd = o3d.io.read_point_cloud(ORIGINAL_CLOUD) # scene 
    # o3d.visualization.draw_geometries([pcd], 'Nube de puntos original')

    # Remove the main planes of the scene to reduce computational load
    scene_pcd = remove_planes_using_ransac(scene_pcd)
 
    # downsample the pcd
    vx_size = 0.005
    scene_pcd = downsample_pcd(scene_pcd, vx_size)
    piggy_pcd = downsample_pcd(piggy_pcd, vx_size)

    # o3d.visualization.draw_geometries([pcd_scene], 'Nube sin planos y con voxel')
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}original_sin_planos.ply", scene_pcd)

    # Compute the keypoints for scene and object
    kp_scene, kp_obj = detect_keypoints_iss(scene_pcd,piggy_pcd)
    # Compute the decriptors for scene keypoints and obj keypoints using FPFH
    scene_desc = descript_fpfh(kp_scene, scene_pcd)
    obj_desc = descript_fpfh(kp_obj, piggy_pcd)
    