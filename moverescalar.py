import open3d as o3d
import numpy as np


POINTCLOUD_DIR = "pointcloud/"
PCD_FILE = "carrapax_ajustado.pcd"



# Cargar la nube de puntos
pcd = o3d.io.read_point_cloud(POINTCLOUD_DIR + PCD_FILE)

# 1️⃣ Convertir los puntos en un array de NumPy
points = np.asarray(pcd.points)

# 2️⃣ Aplicar una traslación en X
y_offset = 400  # Mueve la nube de puntos 2 unidades en X (ajusta según necesites)
x_offset = -150
points[:,0]+= x_offset
points[:, 1] += y_offset


# 3️⃣ Escalar la nube de puntos a la mitad
scale_factor = 0.5  # Factor de escala
points *= scale_factor

# 4️⃣ Actualizar los puntos de la nube
pcd.points = o3d.utility.Vector3dVector(points)

# 5️⃣ Guardar y visualizar la nube modificada

o3d.io.write_point_cloud(POINTCLOUD_DIR+"carrapax_movido.pcd", pcd)
o3d.visualization.draw_geometries([pcd])
