# 🚗 ROI-Based Car Detection using YOLOv8

This project uses **YOLOv8** (Ultralytics) to detect **cars only within a Region of Interest (ROI)** in a video or camera feed. It helps improve focus and computational efficiency by filtering detections spatially.



## 🔍 Key Features

- Real-time or video-based detection using **YOLOv8** (`ultralytics` package)
- **ROI filtering**: Only shows detections where the bounding box centroid lies inside the defined ROI
- Supports:
  - **Webcam** input
  - **Video file** input
  - **Image file** input
- Draws ROI overlay and highlights valid car detections inside it
- Outputs the processed video/image 



## 🧰 Requirements

- Python ≥ 3.8
- [ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy


