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

```bash
python3 tools/build_plan.py \
    --onnx models/yolov8.onnx \
    --engine out/engines/yolov8.plan \
    --precision fp16

# Run the build script
./build.sh

Executables will be installed inside out/install directory
