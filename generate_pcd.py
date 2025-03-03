import open3d as o3d
import os
import numpy as np

MODEL_NAME = "gormiti.stl"

def cargar_y_escalar_modelo(ruta_modelo, altura_objetivo=8.0):
    """
    Carga un modelo 3D y lo escala para que su altura (eje Z) se aproxime a 'altura_objetivo' (por ejemplo, 8.0 cm).
    """
    # Cargar el mesh (soporta STL, OBJ, etc.)
    mesh = o3d.io.read_triangle_mesh(ruta_modelo)
    if mesh.is_empty():
        raise ValueError(f"No se pudo cargar el modelo desde {ruta_modelo}")
    
    # Calcular el bounding box y la altura actual (suponemos que la dimensión Z es la altura)
    bbox = mesh.get_axis_aligned_bounding_box()
    dims = bbox.get_extent()
    altura_actual = dims[2]
    
    # Calcular el factor de escala (evitar división por cero)
    if altura_actual == 0:
        raise ValueError("La altura del modelo es cero. Verifica el modelo 3D.")
    factor_escala = altura_objetivo / altura_actual
    
    # Aplicar la escala
    mesh.scale(factor_escala, center=bbox.get_center())
    return mesh

def muestrear_nube_de_puntos(mesh, num_puntos=20000):
    """
    Muestrea puntos sobre la superficie del mesh usando muestreo Poisson Disk.
    """
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_puntos)
    return pcd

def guardar_nube_de_puntos(pcd, ruta_salida):
    """
    Guarda la nube de puntos en un archivo .pcd.
    """
    if o3d.io.write_point_cloud(ruta_salida, pcd):
        print(f"Nube de puntos guardada exitosamente en: {ruta_salida}")
    else:
        print("Error al guardar la nube de puntos.")

def main():
    # Ruta del modelo 3D del caballo (colócalo en la carpeta 'models')
    ruta_modelo = os.path.join("models", MODEL_NAME)  # o "knight.obj" según tu archivo
    
    # Crear la carpeta de salida si no existe
    carpeta_salida = "pointcloud"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Cargar y escalar el modelo para que tenga una altura de 8.0 (por ejemplo, centímetros)
    try:
        mesh = cargar_y_escalar_modelo(ruta_modelo, altura_objetivo=8.0)
    except Exception as e:
        print(f"Error al cargar/escalar el modelo: {e}")
        return
    
    # Opcional: Visualizar el mesh escalado
    # o3d.visualization.draw_geometries([mesh], window_name="Modelo escalado del Caballo")
    
    # Muestrear la superficie para generar la nube de puntos
    pcd = muestrear_nube_de_puntos(mesh, num_puntos=20000)
    
    # Guardar la nube de puntos en formato .pcd
    ruta_salida = os.path.join(carpeta_salida, "gormiti.pcd")
    guardar_nube_de_puntos(pcd, ruta_salida)
    
    # Opcional: Visualizar la nube de puntos
    # o3d.visualization.draw_geometries([pcd], window_name="Nube de puntos del Caballo")

if __name__ == "__main__":
    main()
