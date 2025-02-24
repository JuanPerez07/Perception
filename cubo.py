import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función para generar puntos en la superficie de un cubo
def generate_cube_points(center, side_length, num_points_per_edge=10):
    half = side_length / 2.0
    # Crear un grid de valores entre -half y half
    grid = np.linspace(-half, half, num_points_per_edge)
    pts = []

    # Caras en x fijo: x = -half y x = half
    for x in [-half, half]:
        for y in grid:
            for z in grid:
                pts.append([x, y, z])
    # Caras en y fijo: y = -half y y = half
    for y in [-half, half]:
        for x in grid:
            for z in grid:
                pts.append([x, y, z])
    # Caras en z fijo: z = -half y z = half
    for z in [-half, half]:
        for x in grid:
            for y in grid:
                pts.append([x, y, z])
    pts = np.array(pts)
    # Desplazar al centro indicado
    pts += np.array(center)
    # Eliminar duplicados (en aristas y vértices se generan solapamientos)
    pts = np.unique(pts, axis=0)
    return pts

# Parámetros globales
# El patrón es 10 veces mayor que el cubo anterior (cubo anterior: lado 10, centro en 0)
# Ahora, la figura general (diamante/octaedro) estará inscrita en un cubo de lado 100 (half = 50)
scale = 50.0  # Valor usado para definir los vértices del diamante (octaedro)
cube_side = 10.0  # Tamaño de cada cubo en cada vértice

# Definir los vértices del diamante (octaedro)
# Usamos 6 vértices: arriba, abajo, derecha, izquierda, frontal y trasero
vertices = np.array([
    [0, 0,  scale],  # Vértice superior
    [0, 0, -scale],  # Vértice inferior
    [scale, 0, 0],   # Vértice derecho
    [-scale, 0, 0],  # Vértice izquierdo
    [0, scale, 0],   # Vértice frontal (hacia arriba en y)
    [0, -scale, 0]   # Vértice trasero (hacia abajo en y)
])

# Definir las conexiones (aristas) entre vértices para formar el octaedro
# Las aristas se definen como pares de índices en el arreglo 'vertices'
edges = [
    (0, 2), (0, 3), (0, 4), (0, 5),   # Desde vértice superior
    (1, 2), (1, 3), (1, 4), (1, 5),   # Desde vértice inferior
    (2, 4), (4, 3), (3, 5), (5, 2)    # Arista que conecta los vértices ecuatoriales (cuadrado)
]

# Generar los puntos de cada cubo (en cada vértice)
cube_points_list = []
for v in vertices:
    pts_cube = generate_cube_points(v, cube_side, num_points_per_edge=10)
    cube_points_list.append(pts_cube)

# Generar puntos para cada arista (línea entre centros de vértices)
edge_points_list = []
edge_resolution = 50  # Cantidad de puntos por arista
for idx1, idx2 in edges:
    p1 = vertices[idx1]
    p2 = vertices[idx2]
    # Generar puntos lineales entre p1 y p2
    line_pts = np.linspace(p1, p2, edge_resolution)
    edge_points_list.append(line_pts)

# Combinar todos los puntos en la nube final
all_points = np.vstack(cube_points_list + edge_points_list)
all_points = np.unique(all_points, axis=0)  # Eliminar duplicados

# Guardar la nube de puntos en un archivo .npy
np.save("diamond_pattern.npy", all_points)
print("La nube de puntos del patrón se ha guardado en 'diamond_pattern.npy'.")

# Visualización en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
           c='blue', marker='o', s=2, alpha=0.7)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Patrón 3D: Diamante con vértices cúbicos y aristas")
plt.show()
