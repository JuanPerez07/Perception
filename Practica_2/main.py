import open3d as o3d  # type: ignore
import numpy as np
import math
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
                distance_threshold=0.03,  # distancia máxima entre un punto y el plano para considerarlo parte de él
                ransac_n=4,               # número de puntos aleatorios usados para estimar un plano
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
    #Obtener distancia para ajustar mejor los radios de ISS
    distance=pcd_scene.compute_nearest_neighbor_distance()
    resolution=np.mean(distance)
    print(f"Resolucion {resolution}")
    # Calculo keypoints escena por ISS
    key_scene = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd_scene,
        salient_radius=0.008,
        non_max_radius=0.0085,
        gamma_21=0.5,
        gamma_32=0.5
    )
    # pintar los keypoints
    key_scene.paint_uniform_color([1, 0, 1])
    pcd_scene.paint_uniform_color([0, 0.5, 0.5])
    o3d.visualization.draw_geometries([key_scene,pcd_scene], 'Keypoints escena')
    #print("Keypoints de la escena detectados")

    # Guardar escena y keypoints
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_downsampled.ply", pcd_scene)
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_keypoints.ply", key_scene)

    # OBJETO: PIGGYBANK
    piggy_pcd = pcd_object
    # Estimar normales
    piggy_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    # Calculo de keypoints por ISS
    key_piggy = o3d.geometry.keypoint.compute_iss_keypoints(
        piggy_pcd,
        salient_radius=0.008,
        non_max_radius=0.008,
        gamma_21=0.5,
        gamma_32=0.5
    )
    #o3d.io.write_point_cloud(f"{OBJ_DIR}piggy_kp_iss.ply", key_piggy)
    

    print("Keypoints detected with ISS for scene and object")

    key_piggy.paint_uniform_color([1, 0, 1])
    piggy_pcd.paint_uniform_color([0, 0.5, 0.5])
    #o3d.visualization.draw_geometries([key_piggy,piggy_pcd],'Key de figura')
    # Return the scene and object keypoints
    return key_scene, key_piggy


"""
Compute the FPFH descriptor for a given pcd and the list of keypoints from ISS
"""
def descript_fpfh(key, pcd, voxel_size=0.005):
    radius_feature = voxel_size * 3  # neighbourhood size

    # Estimar normales si no existen
    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=15))

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

    keypoint_fpfh = fpfh.data[:, indices]
    return keypoint_fpfh

"""
Compute the matching between descriptors using KDTreeFlann and RANSAC
"""
def matching(desc_scene, desc_obj, key_scene, key_obj, max_dist=0.025):
    desc_scene_np = np.asarray(desc_scene).T
    desc_obj_np = np.asarray(desc_obj).T

    kdtree = o3d.geometry.KDTreeFlann(desc_scene_np.T)

    correspondences = []
    nn = 5 # max nearest neighbours searched for matching
    for i, desc in enumerate(desc_obj_np):
        [_, idx, _] = kdtree.search_knn_vector_xd(desc, nn)
        correspondences.append([i, idx[0]])

    corres = o3d.utility.Vector2iVector(correspondences)

    # guardar los matching en un .ply
    exportar_correspondencias_a_obj(key_obj, key_scene, corres)
    # params ajustar correspondencias
    edge_length = 0.45
    normal_angle_thres = math.pi / 8 # algo degrees
    distance_threshold = 0.16
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        key_obj,  # objeto = source
        key_scene,  # escena = target
        corres,
        max_correspondence_distance=max_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(edge_length),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(normal_angle_thres)
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
    o3d.visualization.draw_geometries([kp_obj, kp_scene, line_set])
"""
Guardar matches para visualizar con CloudCompare
"""
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
    piggy_pcd = o3d.io.read_point_cloud(PIGGY) # object
    og_scene_pcd = o3d.io.read_point_cloud(ORIGINAL_CLOUD) # scene 
    # o3d.visualization.draw_geometries([pcd], 'Nube de puntos original')

    # downsample the pcd
    vx_size = 0.005
    scene_pcd = downsample_pcd(og_scene_pcd, vx_size)

    # Remove the main planes of the scene to reduce computational load
    scene_pcd = remove_planes_using_ransac(scene_pcd)
 
    
    #piggy_pcd = downsample_pcd(piggy_pcd, vx_size)

    # o3d.visualization.draw_geometries([pcd_scene], 'Nube sin planos y con voxel')
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}original_sin_planos.ply", scene_pcd)
    #o3d.visualization.draw_geometries([scene_pcd],'Nube de puntos cambiado')

    # Compute the keypoints for scene and object
    kp_scene, kp_obj = detect_keypoints_iss(scene_pcd,piggy_pcd)
    # Compute the decriptors for scene keypoints and obj keypoints using FPFH
    scene_desc = descript_fpfh(kp_scene, scene_pcd)
    print("Descriptors calculated for scene")
    obj_desc = descript_fpfh(kp_obj, piggy_pcd)
    print("Descriptors calculated for object")
    # Realizar matching entre los descriptores usando KDTree junto a RANSAC para filtrar
    match_result = matching(scene_desc, obj_desc, kp_scene, kp_obj) # incluye matriz de transformacion R|t
    print("Matching done with KDTreeFlann and RANSAC")
    # nube de puntos de la escena con el objeto detectado
    insertar_objeto_en_escena(og_scene_pcd, piggy_pcd, match_result.transformation)
    print("Program successfully terminated")
