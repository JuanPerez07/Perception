import open3d as o3d
import numpy as np
import sys
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


POINTCLOUD_DIR = "clouds/scenes/"
CLOUD_NAME = "snap_0point.pcd"
OBJ_NAME = POINTCLOUD_DIR + CLOUD_NAME
pcd = o3d.io.read_point_cloud(OBJ_NAME)
OUTPUT_DIR = "clouds/scenes/"
DEFAULT_EPSILON = 0.1

def remove_planes_using_clustering(pcd, eps=DEFAULT_EPSILON, min_samples=50):
    """
    Removes dominant planes by clustering normals.
    """
    remaining_pcd = pcd
    normals = np.asarray(remaining_pcd.normals)
    
    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    print("Calculated Normals")
    
    # Apply DBSCAN clustering based on normal similarity
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(normals)
    labels, counts = np.unique(clustering.labels_, return_counts=True)
    
    print(f"Clustering and labeling done  with Epsilon = {eps}")
    
    # Sort clusters by size (most frequent = dominant planes)
    dominant_clusters = labels[np.argsort(-counts)]
    
    for cluster in dominant_clusters:
        if cluster == -1:  # Ignore noise points
            continue
        indices = np.where(clustering.labels_ == cluster)[0]
        print(f"Removing plane with {len(indices)} points from cluster {cluster}")
        remaining_pcd = remaining_pcd.select_by_index(indices, invert=True)
    
    print("PCD calculated")
    
    return remaining_pcd


if __name__ == '__main__':
    epsilon_reg  =  False
    epsilon = DEFAULT_EPSILON
    if len(sys.argv) > 1:
        epsilon = float(sys.argv[1])
        epsilon_reg = True

    #o3d.io.write_point_cloud(OUTPUT_DIR + "original.ply", pcd)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=10))

    # Obtener las normales
    norms = np.asarray(pcd.normals)
    # Normalizar para asegurar unidad
    norms /= np.linalg.norm(norms, axis=1, keepdims=True)
    print("Normales calculadas")
    # Aplicar clustering a las normales con DBSCAN para encontrar grupos de orientaciones similares (como en Hough)
    clustering = DBSCAN(eps=epsilon, min_samples=50).fit(norms)
    labels = clustering.labels_
    print("Clustering hecho")
    # Visualizar cada grupo como un posible plano
    max_label = labels.max()
    print(f"Se encontraron {max_label + 1} posibles planos")

    # Colorear la nube según el plano detectado
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = [0, 0, 0, 1]  # Color negro para outliers
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    print("Coloreando nube de puntos")

    #o3d.visualization.draw_geometries([pcd], window_name="Planos detectados (simulación Hough)")

    #o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
    file_name = OUTPUT_DIR + "epsilon_" + str(epsilon) + "_resultados.ply"
    print(f"Saved the coloured pcd named <{file_name}> in {OUTPUT_DIR}")
    o3d.io.write_point_cloud(file_name, pcd)
    o3d.visualization.draw_geometries([pcd], window_name="Filtered with Hough-like Normals")
