# ⚙️ Build & Run Guide

## Dependencies

Make sure the following packages are installed:

- **OpenCV ≥ 4.5.6** (from Ubuntu repositories)
- **CMake ≥ 3.18**
- **TensorRT ≥ 10.0**
- **yaml-cpp**

> CUDA Toolkit & cuDNN are required for TensorRT builds.

---

# Build TensorRT Engine

Before building the framework, generate your TensorRT engine file for YOLO:
Download yolov8n.onnx and put under ./resources/models

```bash
python scripts/build.py \
    --input ./resources/models/yolov8.onnx \
    --output ./resources/models/yolov8.plan \

# Build and run yolo sample
From project directory:
./build.sh
./run.sh

Executables will be installed inside out/install directory
