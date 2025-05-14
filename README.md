# Build Requirements: 
- Project is a CMake project so you must have CMake to configure files.

- Ninja must be installed and its path must be added to the system PATH.

- OpenCV must be installed and its path (especially the binaries) should be included in the PATH.

- CUDA Toolkit must be installed and its environment variables (CUDA_PATH, etc.) should be properly set (usually done by the installer).

- TensorRT must be installed. Its root directory must be manually specified in CMakePresets.json or CMakeLists.txt.

- The project must be built in the x64 Native Tools Command Prompt for VS to ensure all Windows and MSVC dependencies are correctly resolved.

# Model Requirements:
- You must convert yolov8*.onnx file to a TensortRT .plan file format. You can easily create a TensorRT virtual environment in Python and run the converter script in scripts/build.py script. 