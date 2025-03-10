"""
Creates a pointcloud from a STL file using a mesh grid
"""
import open3d as o3d
import os
import numpy as np

MODEL_NAME = "carrapax.stl"
OUTPUT_FILE = "carrapax.pcd"

def load_and_scale_model(path, target_height=8.0):
    """
    Loads 3D (stl) model and scales to a height reference in Z-axis
    """
    # load the mesh (STL, OBJ, etc.)
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Could not load model from {ruta_modelo}")
    
    # Compute bbox & current height (Z-axis)
    bbox = mesh.get_axis_aligned_bounding_box()
    dims = bbox.get_extent()
    current_height = dims[2]
    
    # Calculate scale factor (k)
    if current_height == 0:
        raise ValueError("Model height is zero.")
    k = target_height / current_height
    
    # Apply scale factor
    mesh.scale(k, center=bbox.get_center())
    return mesh

def sample_pcd(mesh, num_points=20000):
    """
    Sample points on the mesh surface using a Poisson Disk.
    """
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    return pcd

def save_pcd(pcd, dir):
    """
    Saves pointcloud in .pcd at the given dir
    """
    if o3d.io.write_point_cloud(dir, pcd):
        print(f"Pointcloud successfully saved at: {dir}")
    else:
        print("Error saving the pointcloud.")

def main():
    # Path to the 3D model
    path = os.path.join("models", MODEL_NAME)  
    
    # Create the output dir if it does not already exist
    output_dir = "pointcloud"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and scale the model
    try:
        mesh = load_and_scale_model(path)
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    # Sample the surface to generate the pcd 
    pcd = sample_pcd(mesh, num_points=20000)
    
    # Save the .pcd 
    out_path = os.path.join(output_dir, OUTPUT_FILE)
    save_pcd(pcd, out_path)
    

if __name__ == "__main__":
    main()
