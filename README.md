# ⚙️ Build & Run Guide

## 📦 Dependencies

Make sure the following packages are installed:

- **OpenCV ≥ 4.5.6** (from Ubuntu repositories)
- **CMake ≥ 3.18**
- **TensorRT ≥ 10.0**
- **yaml-cpp**

> 💡 Optional: CUDA Toolkit & cuDNN are required for TensorRT builds.

---

🧰 Step 1: Build TensorRT Engine

Before building the framework, generate your TensorRT engine file for YOLO:

```bash
python3 tools/build_plan.py \
    --onnx models/yolov8.onnx \
    --engine out/engines/yolov8.plan \
    --precision fp16

🛠 Step 2: Build the Project
# Run the build script
./build.sh

🚀 Step 3: Run the Executables

cd out/install/release/bin

./sort_tracker --video ../../assets/demo.mp4
./yolo --engine ../../out/engines/yolov8.plan
