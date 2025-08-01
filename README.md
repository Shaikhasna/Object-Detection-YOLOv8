﻿# Object-Detection-YOLOv8
# 🎯 Object Detection with YOLOv8

This project implements real-time object detection using YOLOv8 (You Only Look Once). Built with Python and powered by Ultralytics, it allows detection in images, video files, or live webcam feeds.

---

## 🚀 Features

- ⚡ Real-time detection
- 🧠 Uses YOLOv8 pre-trained model
- 🎥 Supports image, video, and webcam input
- 💾 Outputs saved automatically in `/runs/detect/`
- 📦 Custom model support

---

## 🛠️ Tech Stack

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Torch
- Java JDK (for some extended tools like BFG Repo Cleaner)

---

## 📁 Project Structure

agentic-object-detector/
├── detect.py # Main script for detection
├── requirements.txt # Python dependencies
├── yolov8n.pt # YOLOv8 pre-trained weights (nano version)
├── runs/ # Output folder (auto-created)
├── .gitignore
└── README.md


---

## ✅ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shaikhasna/Object-Detection-YOLOv8.git
cd Object-Detection-YOLOv8
```

2.Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (on Windows)

3.Install Python dependencies
pip install -r requirements.txt

4.Download the YOLOv8 model (if not included)
# Download the nano model (small and fast)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

🎬 Usage
Perform real-time object detection using YOLOv8 with various input sources.
🖥️ 3. Run Detection on a Webcam (Real-Time)

python main.py

> Runs live detection using the system's webcam.
> Press q to quit the detection window.








