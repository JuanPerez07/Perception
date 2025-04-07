import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


POINTCLOUD_DIR = "clouds/scenes/"
CLOUD_NAME = "snap_0point.pcd"
OBJ_NAME = POINTCLOUD_DIR + CLOUD_NAME
pcd = o3d.io.read_point_cloud(OBJ_NAME)
OUTPUT_DIR = "clouds/scenes/"
DEFAULT_EPSILON = 0.1


def remove_planes_using_clustering(pcd, eps=DEFAULT_EPSILON, min_samples=50, num_planes=3, spatial_threshold=0.1):
    """
    Removes dominant planes by clustering normals and ensuring the points are within a spatial threshold.
    """
    remaining_pcd = pcd
    normals = np.asarray(remaining_pcd.normals)
    points = np.asarray(remaining_pcd.points)

    # Normalize normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print("Normales calculadas")
    
    # Apply DBSCAN clustering based on normal similarity
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(normals)
    labels, counts = np.unique(clustering.labels_, return_counts=True)
    
    print(f"Clustering done with Epsilon = {eps}")

    dominant_clusters = labels[np.argsort(-counts)]

    all_indices_to_remove = []

    count = 0
    for cluster in dominant_clusters:
        if cluster == -1: continue  # Skip noise
        indices = np.where(clustering.labels_ == cluster)[0]
        
        # Apply a spatial threshold to select the points that belong to this cluster
        cluster_points = points[indices]
        
        print(f"Evaluating Cluster {cluster} with {len(indices)} points...")
        
        # Fit a plane to the cluster points using SVD (Least Squares Method)
        centroid = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal_to_plane = vh[2, :]  # The normal vector is the third row of vh
        
        # Compute the distance of all points to the plane
        distances_to_plane = np.abs(np.dot(centered_points, normal_to_plane))
        
        # Select points that are close to the plane
        indices_to_remove = indices[distances_to_plane < spatial_threshold]
        all_indices_to_remove.extend(indices_to_remove)
        
        print(f"Removing plane {count+1} with {len(indices_to_remove)} points.")
        count += 1
        if count >= num_planes:
            break

    print(f"Total points to remove: {len(all_indices_to_remove)}")

    # If there are indices to remove, perform the filtering
    if len(all_indices_to_remove) > 0:
        filtered_pcd = remaining_pcd.select_by_index(all_indices_to_remove, invert=True)
    else:
        print("No points to remove")
        filtered_pcd = remaining_pcd

    return filtered_pcd


if __name__ == '__main__':
    epsilon = DEFAULT_EPSILON
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=30))

    # Remove planes using clustering
    filtered_pcd = remove_planes_using_clustering(pcd, eps=epsilon, min_samples=80, num_planes=3, spatial_threshold=0.1)
    
    # Save and visualize the filtered cloud
    file_name = OUTPUT_DIR + "filtered.ply"
    o3d.io.write_point_cloud(file_name, filtered_pcd)
    o3d.visualization.draw_geometries([filtered_pcd], window_name="Filtered Point Cloud")
    o3d.visualization.draw_geometries([pcd], window_name="Filtered Point Cloud")
