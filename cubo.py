import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros del cubo
cube_min = -5   # Extremo inferior en cada eje
cube_max = 5    # Extremo superior en cada eje
num_points_per_edge = 20  # Número de puntos por cada dirección en cada cara

# Generar un vector con valores espaciados linealmente entre cube_min y cube_max
grid = np.linspace(cube_min, cube_max, num_points_per_edge)

# Lista para almacenar los puntos
points = []

# Generar puntos para cada una de las 6 caras del cubo

# Cara en x = cube_min
for y in grid:
    for z in grid:
        points.append([cube_min, y, z])

# Cara en x = cube_max
for y in grid:
    for z in grid:
        points.append([cube_max, y, z])

# Cara en y = cube_min
for x in grid:
    for z in grid:
        points.append([x, cube_min, z])

# Cara en y = cube_max
for x in grid:
    for z in grid:
        points.append([x, cube_max, z])

# Cara en z = cube_min
for x in grid:
    for y in grid:
        points.append([x, y, cube_min])

# Cara en z = cube_max
for x in grid:
    for y in grid:
        points.append([x, y, cube_max])

# Convertir la lista a un arreglo de NumPy y eliminar duplicados (en aristas y vértices se generan puntos repetidos)
points = np.array(points)
points = np.unique(points, axis=0)

# Guardar la nube de puntos en un archivo .npy
np.save("cubo_point_cloud.npy", points)
print("La nube de puntos se ha guardado en 'cubo_point_cloud.npy'.")

# Visualización de la nube de puntos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', s=5)
ax.set_xlabel("Eje X")
ax.set_ylabel("Eje Y")
ax.set_zlabel("Eje Z")
ax.set_title("Nube de puntos 3D - Cubo")
plt.show()
