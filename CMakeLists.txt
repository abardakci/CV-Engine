cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------
# Find external packages
# ----------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# ----------------------------------------
# TensorRT package (manually)
# ----------------------------------------
set(TRT_INCLUDES /usr/include/x86_64-linux-gnu)
set(TRT_LIBS
    /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10.10.0
)
include_directories(/usr/local/cuda-12.8/targets/x86_64-linux/include/)

# ----------------------------------------
# Subprojects
# ----------------------------------------

add_subdirectory(core/common)     # core_common
add_subdirectory(core/cv_impl)    # core_cv
add_subdirectory(core/algorithms) # core_algo
add_subdirectory(core/nn_backend) # core_engine

# add_subdirectory(core/tracking) # core_tracker
add_subdirectory(core/inference)  # core_infer

# ----------------------------------------
# Executable target: Object Detector
# ----------------------------------------
add_executable(yolo app/detect.cpp)

target_link_libraries(yolo PUBLIC
    core_common
    core_cv
    core_algo
    core_infer
    core_engine
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TRT_LIBS}
    yaml-cpp
)

target_include_directories(yolo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TRT_INCLUDES}
)

# Install executable
install(TARGETS yolo RUNTIME                     DESTINATION ${CMAKE_SOURCE_DIR}/install)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/config DESTINATION ${CMAKE_SOURCE_DIR}/install)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/assets DESTINATION ${CMAKE_SOURCE_DIR}/install)

# ----------------------------------------
# Executable target: Object Tracker
# ----------------------------------------
# add_executable(Tracker app/track.cpp)

# target_link_libraries(Tracker PUBLIC
#     core_common
#     core_cv
#     core_algo
#     core_tracker
#     ${OpenCV_LIBS}
#     CUDA::cudart
#     ${TRT_LIBS}
# )

# target_include_directories(Tracker PRIVATE
#     ${OpenCV_INCLUDE_DIRS}
#     ${TRT_INCLUDES}
# )
