# Proyecto de Visión 3D y Detección de Objetos 🤖📷

Este repositorio agrupa dos prácticas independientes:

- **Practica_1**: Calibración de cámara y proyección de nubes de puntos.
- **Practica_2**: Detección y alineación de objetos 3D mediante un pipeline tradicional.

---

## Estructura del proyecto 📂

```
📦 .
├── Practica_1/
│   ├── main.py
│   ├── generate_pcd.py
│   ├── create_dataset.py
│   ├── figure3D.py
│   ├── params/
│   ├── dataset/
│   ├── pointcloud/
│   ├── mre_comparison/
│   └── (otros scripts y recursos para calibración)
└── Practica_2/
    ├── main.py           # Pipeline de detección y alineación
    ├── charizard.py      # Segmentación 3D de Charmander + pipeline completo
    └── datos/            # Modelos y nubes de puntos de escena y objetos
```

---

## Practica_1: Calibración de Cámara 📷🔍

Este módulo permite calibrar una cámara con un patrón de ajedrez y proyectar nubes de puntos en vídeo en tiempo real.

### 1. Calibración de Cámara  
```bash
python main.py
```
Guarda los parámetros en `params/`.

### 2. Proyección 3D en Vídeo  
```bash
python figure3D.py
```

### 3. Análisis Experimental  
- Carpeta `mre_comparison/` con estudio de **Mean Reprojection Error** para distintos calibrados.

---

## Practica_2: Detección y Alineación de Objetos 🛠️☁️

Objetivo: hallar la transformación rígida que alinea la nube de puntos de un objeto dentro de una escena.

### 1. Reducción de la nube de puntos  
- Downsampling para disminuir densidad y coste computacional.  
- Elimina redundancia (puntos cercanos) y mejora estabilidad de la convergencia.

### 2. Eliminación de planos dominantes  
- Segmenta y descarta grandes planos (p. ej. mesas) usando un umbral de distancia.  
- [Insertar imágenes antes/después de la eliminación de planos]

### 3. Detección de puntos de interés  
- Algoritmo **ISS** (Open3D) sobre escena y objeto:  
  - Estimación de normales vía KD-Tree + KNN.  
  - Parámetros clave:  
    - `salient_radius`, `non_max_radius`, `gamma_21`, `gamma_32`  
  - Visualización de keypoints coloreados.

### 4. Cálculo de descriptores FPFH  
- Calcula FPFH en radio y vecindad definidos (requiere normales previas).  
- Extrae descriptor sólo en los keypoints usando KD-TreeFlann.

### 5. Emparejamiento inicial y RANSAC  
- Correspondencias con `search_knn_vector_xd`.  
- Filtrado RANSAC sobre correspondencias:  
  - Umbrales: `edge_length`, `distance`, `normal_angle`.  
- Obtiene **transformación rígida** inicial basada en inliers.

### 6. Refinamiento con ICP  
- Usa la transformación de RANSAC como punto de partida.  
- Algoritmo **ICP punto-a-plano** (Open3D) para minimizar error iterativamente.  
- Parámetros: umbral de distancia, límite de iteraciones, criterio de convergencia.

---

## Cómo usar Practica_2 🚀

```bash
cd Practica_2
# Alineación genérica
python main.py --scene datos/escena.pcd --object datos/objeto.pcd

# Segmentación de Charmander + pipeline completo
python charizard.py --scene datos/escena_con_charizard.pcd --model datos/charmander.pcd
```

---

## Dependencias 🛠

```bash
pip install numpy opencv-python open3d
```

---

## Autores 📝

- **Aitor Ruiz Bautista**  
- **Juan Pérez Fernández**
