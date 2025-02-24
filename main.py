import cv2
import numpy as np
import glob
# Dataset dir and img format
DATA_DIR = "dataset"
IMG_FORMAT = "/*.png"
# Tamaño del patrón de ajedrez (número de esquinas internas por fila y columna)
CHESSBOARD_SIZE = (6, 9)  # Ajusta según tu patrón
SQUARE_SIZE = 29  # Tamaño de cada cuadro en mm (ajustar según tu patrón)

# Criterios de precisión para el algoritmo de detección de esquinas
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Puntos 3D en el espacio real del ajedrez (coordenadas en milímetros)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Listas para almacenar puntos 3D (mundo real) y puntos 2D (imagen)
objpoints = []  # Puntos 3D
imgpoints = []  # Puntos 2D detectados en las imágenes

# Cargar todas las imágenes de la carpeta "datasheet/"
images = glob.glob(DATA_DIR + IMG_FORMAT)

img = cv2.imread(images[0])  # Leer la primera imagen
height, width = img.shape[:2]  # Obtener dimensiones
image_shape = (width, height)  # OpenCV usa (ancho, alto)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buscar esquinas del patrón de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:  # Si se encontraron las esquinas
        objpoints.append(objp)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(refined_corners)

        # Dibujar las esquinas detectadas
        #cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, refined_corners, ret)
        #cv2.imshow('Detected Chessboard', img)
        #cv2.waitKey(0)
# Calibrar la cámara
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_shape, None, None
)

# Mostrar los resultados
print("📷 Matriz de calibración (K):")
print(camera_matrix)

print("\n🎯 Coeficientes de distorsión:")
print(dist_coeffs)

print("\n🔁 Vectores de rotación (uno por imagen):")
print(rvecs)

print("\n📍 Vectores de traslación (uno por imagen):")
print(tvecs)


cv2.destroyAllWindows()