# Practica 1: Camera Calibration 📷🔍

This project performs **camera calibration** using the chessboard pattern and **point cloud projection** for 3D vision applications. It includes tools for dataset collection, calibration processing, point cloud generation, and evaluation of calibration results.

---

## **Project Structure** 📂

```
📦 Practica_1
├── main.py                 # Main script for camera calibration
├── generate_pcd.py         # Converts STL model surface into a point cloud (.pcd)
├── create_dataset.py       # Captures images of the calibration pattern using a webcam
├── figure3D.py             # Projects two point clouds onto the calibration pattern in live video
├── params/                 # Stores camera calibration parameters
├── dataset/                # Contains images used for the calibration process
├── pointcloud/             # Contains .pcd files for experimentation
├── mre_comparison/         # Stores results comparing different camera calibrations
└── (Other files)           # Used to scale and transform raw point clouds from STL models
```

---

## **Usage** 🚀

### **1) Camera Calibration**
Run `main.py` to calibrate the camera and save the calibration parameters:
```bash
python main.py
```
- The results will be stored in the `params/` directory.

### **2) Visualizing 3D Projections in Live Video**
Run `figure3D.py` to project point clouds onto the calibration pattern in real time:
```bash
python figure3D.py
```

---

## **Experimental Results** 📊
- The **mre_comparison/** folder contains **Mean Reprojection Error (MRE) analysis** comparing different calibration results.
- Raw point clouds extracted from STL models were **scaled and transformed** for improved accuracy.

---

## **Requirements** 🛠
Ensure you have the required Python packages installed:
```bash
pip install numpy opencv-python open3d
```

---

## **Authors** 📝
Developed by **Aitor Ruiz Bautista** and **Juan Pérez Fernández**.
