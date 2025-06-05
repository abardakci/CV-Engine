### Build Requirements
- This is a CMake-based project. You must have CMake installed for configuration.

- Ninja must be installed, and its path should be added to your PATH.

- OpenCV must be installed, and its binaries should be accessible via PATH.

- CUDA Toolkit must be installed, with environment variables (e.g., CUDA_PATH) properly set (usually handled by the installer).

- TensorRT must be installed. You need to specify its root directory to CMake.

- Build the project using the x64 Native Tools Command Prompt for Visual Studio to ensure all Windows and MSVC dependencies are correctly resolved.

### Model Requirements
- Convert the yolov8.onnx model file to TensorRT .plan format.

- You can create a Python virtual environment for TensorRT and run the converter script located at scripts/build.py.
