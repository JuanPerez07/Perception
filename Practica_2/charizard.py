import open3d as o3d  # type: ignore
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque # save clusters
import os # hanlde directories

# DIRs
CHARMANDER_SOURCE = "charmander_obj/pcd_9.pcd"
CHARMANDER_SCENE = "clutter_scene/pcd_26.pcd"
OUTPUT_DIR = "clutter_scene/"
PLANOS = 3
"""
Remove planes using RANSAC
"""
def remove_planes_using_ransac(pcd, threshold=0.03):
    for i in range(PLANOS):
        _, inliers = pcd.segment_plane(
                distance_threshold=threshold,  # distancia máxima entre un punto y el plano para considerarlo parte de él
                ransac_n=3,               # número de puntos aleatorios usados para estimar un plano
                num_iterations=1000       # número de iteraciones para encontrar el mejor plano
        )
        # keep only the outliers
        pcd = pcd.select_by_index(inliers, invert=True)
        # save the pcd without the inliers (only outliers)
        #o3d.io.write_point_cloud(f"{OUTPUT_DIR}step_ransac_{i}.ply", pcd)
        
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
        non_max_radius=0.0085,
        gamma_21=0.5,
        gamma_32=0.5
    )
    # pintar los keypoints
    key_scene.paint_uniform_color([1, 0, 1])
    pcd_scene.paint_uniform_color([0, 0.5, 0.5])
    #o3d.visualization.draw_geometries([key_scene,pcd_scene], 'Keypoints escena')
    #print("Keypoints de la escena detectados")

    # Guardar escena y keypoints
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_downsampled.ply", pcd_scene)
    #o3d.io.write_point_cloud(f"{OUTPUT_DIR}scene_keypoints.ply", key_scene)

    # OBJETO: PIGGYBANK
    piggy_pcd = pcd_object
    # Estimar normales
    piggy_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    
    """ Calculo de keypoints para piggy 
    key_piggy = o3d.geometry.keypoint.compute_iss_keypoints(
        piggy_pcd,
        salient_radius=0.008,
        non_max_radius=0.008,
        gamma_21=0.5,
        gamma_32=0.5
    )
    """
    key_piggy = o3d.geometry.keypoint.compute_iss_keypoints(
        piggy_pcd,
        salient_radius=0.008,#radio de vecindad
        non_max_radius=0.0055,#filtro para que no estén super cerca
        gamma_21=0.75,#cambios en la curvatura
        gamma_32=0.75#cambio de curvatura en otra direccion
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
    #o3d.visualization.draw_geometries([kp_obj, kp_scene, line_set])
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
    #o3d.visualization.draw_geometries([escena_completa],'Final')
    #return escena_completa

def es_primitiva(cluster, plane_ratio=0.9, plano_thresh=0.01, linealidad_thresh=0.95):
    """
    Devuelve True si el cluster representa una forma geométrica simple (plano o línea).
    """
    try:
        plane_model, inliers = cluster.segment_plane(distance_threshold=plano_thresh,
                                                     ransac_n=3,
                                                     num_iterations=1000)
        if len(inliers) / len(cluster.points) > plane_ratio:
            return True  # Se ajusta fuertemente a un plano
    except:
        pass

    # 2. Verificar si los puntos son casi lineales usando PCA
    pts = np.asarray(cluster.points)
    if pts.shape[0] < 3:
        return True

    centered = pts - np.mean(pts, axis=0)
    cov = np.cov(centered.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]

    if eigvals[1] / eigvals[0] < (1 - linealidad_thresh):
        return True  # Segunda componente muy pequeña → estructura lineal

    return False  # No parece primitiva

def regionGrowth(pcd, cluster_size=200, angle_threshold_deg=30, distance_threshold=0.02):
    """
    Segmentación por crecimiento de regiones basada en normales.
    
    Parámetros:
    - pcd_path: ruta al archivo .pcd
    - angle_threshold_deg: umbral en grados para diferencia de normales
    - distance_threshold: distancia máxima entre vecinos
    
    Retorna:
    - Lista de clusters (cada uno es un open3d.geometry.PointCloud)
    """
    # Estimar las normales
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.normalize_normals()

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n_points = len(points)
    
    angle_threshold = np.cos(np.deg2rad(angle_threshold_deg))  # cos(angle)

    # KD-tree para buscar vecinos
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    visited = np.zeros(n_points, dtype=bool)
    clusters = []

    for i in range(n_points):
        if visited[i]:
            continue

        queue = deque()
        queue.append(i)
        visited[i] = True
        cluster_indices = [i]

        while queue:
            current = queue.popleft()
            [_, idxs, dists] = kdtree.search_radius_vector_3d(pcd.points[current], distance_threshold)
            for j in idxs:
                if not visited[j]:
                    # Comparar normales
                    dot = np.dot(normals[current], normals[j])
                    if dot > angle_threshold:
                        visited[j] = True
                        queue.append(j)
                        cluster_indices.append(j)

        # Si el cluster es suficientemente grande
        if len(cluster_indices) > cluster_size:
            cluster = pcd.select_by_index(cluster_indices)
            clusters.append(cluster)
    
    assert(len(clusters) != 0)
    
    clusters_filtered = []
    for c in clusters:
        if not es_primitiva(c):
            clusters_filtered.append(c)
    
    assert(len(clusters_filtered) != 0)
    
    return clusters_filtered

"""
Segmentar objeto 3D
"""
def segmentObj(source_pcd):
    # downsample pcd and remove planes first
    obj = downsample_pcd(source_pcd, vx_size=0.01)
    obj = remove_planes_using_ransac(obj)


    clusters = regionGrowth(obj)
#   Crear carpeta si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Crear una lista para juntar todos los puntos coloreados
    combined_points = []
    combined_colors = []

    for i, cluster in enumerate(clusters):
        # Color aleatorio por cluster
        color = np.random.rand(3)
        np_cluster_points = np.asarray(cluster.points)
        np_colors = np.tile(color, (len(np_cluster_points), 1))

        combined_points.append(np_cluster_points)
        combined_colors.append(np_colors)

    # Combinar en una sola nube
    if combined_points:
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)

        o3d.io.write_point_cloud(f"{OUTPUT_DIR}clusters_combinados.ply", combined_pcd)
        print(f"Guardado: {OUTPUT_DIR}clusters_combinados.ply")

    # Guardar la nube sin planos, posiblemente útil para depuración
    o3d.io.write_point_cloud(f"{OUTPUT_DIR}obj_segmentado.ply", obj)
    return obj

if __name__ == '__main__':
    # load both scene and objects pcds
    obj_pcd = o3d.io.read_point_cloud(CHARMANDER_SOURCE) # object_scene
    # segment the target obj
    obj_segmented = segmentObj(obj_pcd) # segment the 3D obj
    quit()
    og_scene_pcd = o3d.io.read_point_cloud(CHARMANDER_SCENE) # scene 
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
    kp_scene, kp_obj = detect_keypoints_iss(scene_pcd,obj_segmented)
    # Compute the decriptors for scene keypoints and obj keypoints using FPFH
    scene_desc = descript_fpfh(kp_scene, scene_pcd)
    print("Descriptors calculated for scene")
    obj_desc = descript_fpfh(kp_obj, obj_segmented)
    print("Descriptors calculated for object")
    # Realizar matching entre los descriptores usando KDTree junto a RANSAC para filtrar
    match_result = matching(scene_desc, obj_desc, kp_scene, kp_obj) # incluye matriz de transformacion R|t
    print("Matching done with KDTreeFlann and RANSAC")
    # nube de puntos de la escena con el objeto detectado
    insertar_objeto_en_escena(og_scene_pcd, obj_segmented, match_result.transformation)
    print("Program successfully terminated")
