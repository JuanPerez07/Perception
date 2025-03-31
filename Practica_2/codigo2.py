import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

POINTCLOUD_DIR = "clouds/scenes/"
CLOUD_NAME = "snap_0point.pcd"
OBJ_NAME = POINTCLOUD_DIR + CLOUD_NAME
pcd = o3d.io.read_point_cloud(OBJ_NAME)
OUTPUT_DIR = "clouds/scenes/"

def remove_planes_using_clustering(pcd, eps=0.1, min_samples=50, max_planes=5):
    """
    Removes dominant planes by clustering normals.
    """
    remaining_pcd = pcd
    normals = np.asarray(remaining_pcd.normals)
    
    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Apply DBSCAN clustering based on normal similarity
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(normals)
    labels, counts = np.unique(clustering.labels_, return_counts=True)
    
    # Sort clusters by size (most frequent = dominant planes)
    dominant_clusters = labels[np.argsort(-counts)]
    
    for cluster in dominant_clusters[:max_planes]:
        if cluster == -1:  # Ignore noise points
            continue
        indices = np.where(clustering.labels_ == cluster)[0]
        print(f"Removing plane with {len(indices)} points from cluster {cluster}")
        remaining_pcd = remaining_pcd.select_by_index(indices, invert=True)
    
    return remaining_pcd

o3d.io.write_point_cloud(OUTPUT_DIR + "original.ply", pcd)

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

# Apply clustering-based plane removal
filtered_pcd = remove_planes_using_clustering(pcd)

o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
o3d.io.write_point_cloud(OUTPUT_DIR + "resultados.ply", filtered_pcd)