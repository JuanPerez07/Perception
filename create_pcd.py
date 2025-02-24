import numpy as np

# Función para generar puntos en la superficie de un cubo
def generate_cube_points(center, side_length, num_points_per_edge=10):
    half = side_length / 2.0
    grid = np.linspace(-half, half, num_points_per_edge)
    pts = []

    # Caras en x fijo
    for x in [-half, half]:
        for y in grid:
            for z in grid:
                pts.append([x, y, z])

    # Caras en y fijo
    for y in [-half, half]:
        for x in grid:
            for z in grid:
                pts.append([x, y, z])

    # Caras en z fijo
    for z in [-half, half]:
        for x in grid:
            for y in grid:
                pts.append([x, y, z])

    pts = np.array(pts)
    pts += np.array(center)  # Desplazar al centro indicado
    return np.unique(pts, axis=0)  # Eliminar duplicados

# Parámetros globales
scale = 50.0  # Tamaño general del octaedro
cube_side = 10.0  # Tamaño de los cubos en los vértices

# Definir los vértices del diamante (octaedro)
vertices = np.array([
    [0, 0,  scale],  # Vértice superior
    [0, 0, -scale],  # Vértice inferior
    [scale, 0, 0],   # Vértice derecho
    [-scale, 0, 0],  # Vértice izquierdo
    [0, scale, 0],   # Vértice frontal
    [0, -scale, 0]   # Vértice trasero
])

# Definir las conexiones entre vértices (aristas)
edges = [
    (0, 2), (0, 3), (0, 4), (0, 5),   # Desde el vértice superior
    (1, 2), (1, 3), (1, 4), (1, 5),   # Desde el vértice inferior
    (2, 4), (4, 3), (3, 5), (5, 2)    # Conexiones ecuatoriales
]

# Generar puntos de cada cubo en los vértices
cube_points_list = [generate_cube_points(v, cube_side) for v in vertices]

# Generar puntos para las aristas
edge_points_list = []
edge_resolution = 50  # Cantidad de puntos por arista
for idx1, idx2 in edges:
    p1, p2 = vertices[idx1], vertices[idx2]
    line_pts = np.linspace(p1, p2, edge_resolution)
    edge_points_list.append(line_pts)

# Combinar todos los puntos en una única nube de puntos
all_points = np.vstack(cube_points_list + edge_points_list)
all_points = np.unique(all_points, axis=0)  # Eliminar duplicados
num_points = all_points.shape[0]

# Guardar en formato PCD
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
        f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

print(f"La nube de puntos del patrón se ha guardado en '{pcd_filename}'.")
