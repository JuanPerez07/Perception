import open3d as o3d  # type: ignore
import numpy as np
import math
import matplotlib.pyplot as plt

# DIR SCENES
POINTCLOUD_DIR = "clouds/scenes/"
SCENE_NAME = "snap_0point.pcd"
PEPPER_POINTCLOUD_DIR="pepper_scene/"
PEPPER_SCENE_NAME="pcd_21.pcd"
ORIGINAL_CLOUD = POINTCLOUD_DIR + SCENE_NAME
OUTPUT_DIR = "clouds/scenes/"
PEPPER_POINTCLOUD=PEPPER_POINTCLOUD_DIR+PEPPER_SCENE_NAME

# DIR OBJETOS
OBJ_DIR = "clouds/objects/"
MUG_NAME = "s0_mug_corr.pcd"
PIGGY_NAME = "s0_piggybank_corr.pcd"
PLANT_NAME = "s0_plant_corr.pcd"
PLC_NAME = "s0_plc_corr.pcd"
PEPPER_NAME_OBJ="pcd_33.pcd"

# OBJ PCDS NAMES DIR
MUG = OBJ_DIR + MUG_NAME
PIGGY = OBJ_DIR + PIGGY_NAME
PLANT = OBJ_DIR + PLANT_NAME
PLC = OBJ_DIR + PLC_NAME
OBJETOS = [MUG, PIGGY, PLANT, PLC]
PEPPER="pepper_obj/"+PEPPER_NAME_OBJ

# PARTE ADICIONAL
CHARMANDER_SOURCE = "charmander_obj/pcd_9.pcd"
CHARMANDER_SCENE = "clutter_scene/pcd_26.pcd"

PLANOS = 3
"""
Remove planes using RANSAC
"""
def remove_planes_using_ransac(pcd):
    threshold=0.03
    for i in range(PLANOS):
        
        # detect a plane in the cloud
        if i==2:
            threshold=0.02
        else:
            threshold=0.03
        _, inliers = pcd.segment_plane(
                distance_threshold=threshold,  # distancia máxima entre un punto y el plano para considerarlo parte de él
                ransac_n=3,               # número de puntos aleatorios usados para estimar un plano
                num_iterations=1000       # número de iteraciones para encontrar el mejor plano
        )
        # keep only the outliers
        pcd = pcd.select_by_index(inliers, invert=True)
        # save the pcd without the inliers (only outliers)
        o3d.io.write_point_cloud(f"{OUTPUT_DIR}step_ransac_{i}.ply", pcd)
        
    #o3d.visualization.draw_geometries([pcd],'Ver')
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
        salient_radius=0.008,
        non_max_radius=0.0065,
        gamma_21=0.45,
        gamma_32=0.45
    )
    # pintar los keypoints
    key_scene.paint_uniform_color([1, 0, 1])
    pcd_scene.paint_uniform_color([0, 0.5, 0.5])
    #o3d.visualization.draw_geometries([key_scene,pcd_scene], 'Keypoints escena')
    #print("Keypoints de la escena detectados")

    # Guardar escena y keypoints
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_downsampled.ply", pcd_scene)
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_keypoints.ply", key_scene)

   
    
    # Estimar normales
    pcd_object.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
   
    key_obj = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_object,
        salient_radius=0.008,#radio de vecindad
        non_max_radius=0.006,#filtro para que no estén super cerca
        gamma_21=0.6,#cambios en la curvatura
        gamma_32=0.6,#cambio de curvatura en otra direccion
        min_neighbors=4
    )
    #o3d.io.write_point_cloud(f"{OBJ_DIR}piggy_kp_iss.ply", key_piggy)
    

    print("Keypoints detected with ISS for scene and object")

    key_obj.paint_uniform_color([1, 0, 1])
    pcd_object.paint_uniform_color([0, 0.5, 0.5])
    #o3d.visualization.draw_geometries([key_obj,obj_pcd],'Key de figura')
    # Return the scene and object keypoints
    return key_scene, key_obj


"""
Compute the FPFH descriptor for a given pcd and the list of keypoints from ISS
"""
def descript_fpfh(key, pcd, voxel_size=0.005):
    radius_feature = voxel_size * 3  # neighbourhood size

    # Estimar normales si no existen
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

    # Calcular FPFH para toda la nube
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=15)
    )

    # Para cada punto en keypoints, encontrar su índice más cercano en pcd
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    indices = []
    for kp in key.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(kp, 1)
        indices.append(idx[0])

    data_all=np.asarray(fpfh.data)
    filtered_data=data_all[:,indices]
    fpfh_keypoints=o3d.pipelines.registration.Feature()
    fpfh_keypoints.data=filtered_data
   
   
    return fpfh_keypoints

"""
Compute the matching between descriptors using KDTreeFlann and RANSAC
"""
def matching(keypoints_scene, keypoints_obj, desc_scene, desc_obj,vx_size=0.025):


    max_dist=vx_size*1.5
    angle=np.deg2rad(38)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=keypoints_obj,
        target=keypoints_scene,
        source_feature=desc_obj,
        target_feature=desc_scene,
        mutual_filter=True,  # Para que las correspondencias sean mutuas
        max_correspondence_distance=vx_size,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=5,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.65),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.16),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(angle)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    return result
"""
Visualizar los mathcing realizados
"""
def visualizar_correspondencias(kp_obj, kp_scene, correspondences):
    # Colores distintos para objeto y escena
    kp_obj.paint_uniform_color([1, 0, 0])  # rojo
    kp_scene.paint_uniform_color([0, 1, 0])  # verde

    # Crear una geometría de líneas entre keypoints emparejados
    lines = []
    colors = []
    points = []

    obj_points = np.asarray(kp_obj.points)
    scene_points = np.asarray(kp_scene.points)

    for obj_idx, scene_idx in np.asarray(correspondences):
        p1 = obj_points[obj_idx]
        p2 = scene_points[scene_idx]
        points.append(p1)
        points.append(p2)
        lines.append([len(points)-2, len(points)-1])
        colors.append([0, 0, 1])  # azul para las líneas

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Mostrar
    #o3d.visualization.draw_geometries([kp_obj, kp_scene, line_set])
"""
Guardar matches para visualizar con CloudCompare
"""

def refine_registration_icp(source, target, init_transform, voxel_size=0.005):
    distance_threshold = voxel_size *1.5 # o prueba con voxel_size * 2
    
    icp_result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=distance_threshold,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    return icp_result
def exportar_correspondencias_a_obj(kp_obj, kp_scene, correspondences, file_path=OUTPUT_DIR+"correspondencias_lines.obj"):
    obj_points = np.asarray(kp_obj.points)
    scene_points = np.asarray(kp_scene.points)

    with open(file_path, "w") as f:
        vertices = []
        for obj_idx, scene_idx in np.asarray(correspondences):
            if obj_idx < len(obj_points) and scene_idx < len(scene_points):
                p1 = obj_points[obj_idx]
                p2 = scene_points[scene_idx]
                vertices.append(p1)
                vertices.append(p2)

        # Escribir vértices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Escribir líneas entre cada par consecutivo
        for i in range(0, len(vertices), 2):
            f.write(f"l {i + 1} {i + 2}\n")

"""
Insertar en la nube de puntos de la escena el objeto usando la matriz R | t
"""
def insertar_objeto_en_escena(scene_pcd, obj_pcd, transformation_matrix):

    # Hacemos una copia del objeto para no modificar el original
    obj_transformado = obj_pcd.transform(transformation_matrix.copy())

    obj_transformado.paint_uniform_color([1 ,0,1])
    # Combinar ambas nubes
    escena_completa = scene_pcd + obj_transformado

    # Guardamos o retornamos la nube combinada
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}objeto_inyectado_en_escena.ply", escena_completa)
    o3d.visualization.draw_geometries([escena_completa],'Final')
    #return escena_completa

if __name__ == '__main__':
    # load both scene and objects pcds
    obj_pcd = o3d.io.read_point_cloud(PIGGY) # object
    og_scene_pcd = o3d.io.read_point_cloud(ORIGINAL_CLOUD) # scene 
    # o3d.visualization.draw_geometries([pcd], 'Nube de puntos original')

    # downsample the pcd
    vx_size = 0.005
    scene_pcd = downsample_pcd(og_scene_pcd, vx_size)


    # Remove the main planes of the scene to reduce computational load
    scene_pcd = remove_planes_using_ransac(scene_pcd)
 
    
   

    # o3d.visualization.draw_geometries([pcd_scene], 'Nube sin planos y con voxel')
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}original_sin_planos.ply", scene_pcd)
    #o3d.visualization.draw_geometries([scene_pcd],'Nube de puntos cambiado')

    # Compute the keypoints for scene and object
    kp_scene, kp_obj = detect_keypoints_iss(scene_pcd,obj_pcd)
    # Compute the decriptors for scene keypoints and obj keypoints using FPFH
    scene_desc = descript_fpfh(kp_scene, scene_pcd)
    print("Descriptors calculated for scene")
    obj_desc = descript_fpfh(kp_obj, obj_pcd)
    print("Descriptors calculated for object")
    # Realizar matching entre los descriptores usando KDTree junto a RANSAC para filtrar
    match_result = matching(kp_scene,kp_obj,scene_desc,obj_desc) # incluye matriz de transformacion R|t
    print("Matching done with KDTreeFlann and RANSAC")
    refined=refine_registration_icp(obj_pcd,scene_pcd,match_result.transformation,vx_size)
    # nube de puntos de la escena con el objeto detectado
    insertar_objeto_en_escena(og_scene_pcd, obj_pcd, refined.transformation)
    print("Program successfully terminated")
