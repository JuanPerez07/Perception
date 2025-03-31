import numpy as np

# Function to generate points on the surface of a cube
def generate_cube_points(center, side_length, num_points_per_edge=10):
    half = side_length / 2.0
    grid = np.linspace(-half, half, num_points_per_edge)  # Generate evenly spaced points along the edges
    pts = []

    # Faces with fixed x-coordinates
    for x in [-half, half]:
        for y in grid:
            for z in grid:
                pts.append([x, y, z])

    # Faces with fixed y-coordinates
    for y in [-half, half]:
        for x in grid:
            for z in grid:
                pts.append([x, y, z])

    # Faces with fixed z-coordinates
    for z in [-half, half]:
        for x in grid:
            for y in grid:
                pts.append([x, y, z])

    pts = np.array(pts)
    pts += np.array(center)  # Shift the points to the specified center
    return np.unique(pts, axis=0)  # Remove duplicate points

# Global parameters
scale = 50.0  # Overall size of the octahedron
cube_side = 10.0  # Size of the cubes at the vertices

# Define the vertices of the octahedron
vertices = np.array([
    [0, 0,  scale],  # Top vertex
    [0, 0, -scale],  # Bottom vertex
    [scale, 0, 0],   # Right vertex
    [-scale, 0, 0],  # Left vertex
    [0, scale, 0],   # Front vertex
    [0, -scale, 0]   # Back vertex
])

# Define the edges between vertices
edges = [
    (0, 2), (0, 3), (0, 4), (0, 5),   # Edges from the top vertex
    (1, 2), (1, 3), (1, 4), (1, 5),   # Edges from the bottom vertex
    (2, 4), (4, 3), (3, 5), (5, 2)    # Equatorial connections
]

# Generate points for each cube at the vertices
cube_points_list = [generate_cube_points(v, cube_side) for v in vertices]

# Generate points along the edges
edge_points_list = []
edge_resolution = 50  # Number of points per edge
for idx1, idx2 in edges:
    p1, p2 = vertices[idx1], vertices[idx2]
    line_pts = np.linspace(p1, p2, edge_resolution)  # Interpolate points along the edge
    edge_points_list.append(line_pts)

# Combine all points into a single point cloud
all_points = np.vstack(cube_points_list + edge_points_list)
all_points = np.unique(all_points, axis=0)  # Remove duplicate points
num_points = all_points.shape[0]  # Get the total number of points

# Save the point cloud in PCD format
pcd_header = f"""# .PCD v.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""

pcd_filename = "diamond_pattern.pcd"
with open(pcd_filename, "w") as f:
    f.write(pcd_header)
    for pt in all_points:
        f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")  # Write each point to the file

print(f"The point cloud of the pattern has been saved in '{pcd_filename}'.")
