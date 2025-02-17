import cv2
import os

carpeta = "datasheet"

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Contador para los nombres de archivo
contador = 1

# Crear una ventana
cv2.namedWindow("Presiona 'c' para capturar - 'q' para salir")

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mostrar el frame en la ventana
    cv2.imshow("Presiona 'c' para capturar - 'q' para salir", frame)

    # Capturar la tecla presionada
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Si se presiona 'c'
        filename = os.path.join(carpeta, f"{contador}.png")  # Ruta completa
        cv2.imwrite(filename, frame)  # Guardar la imagen
        print(f"Imagen guardada como {filename}")
        contador += 1  # Incrementar el contador

    elif key == ord('q'):  # Si se presiona 'q', salir
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
