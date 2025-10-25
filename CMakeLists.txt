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

# ----------------------------------------
# Subprojects
# ----------------------------------------
add_subdirectory(core/common)      # core_common
add_subdirectory(core/cv_wrapper)  # core_cv
add_subdirectory(core/algorithms)  # core_algorithms
add_subdirectory(core/trt_backend) # core_trt_backend
add_subdirectory(core/inference)   # core_infer
add_subdirectory(core/tracker)     # core_tracker
add_subdirectory(core/interfaces)  # core_interfaces

# ----------------------------------------
# Executable target: Object Detector
# ----------------------------------------
add_executable(yolo app/detect.cpp)

target_link_libraries(yolo PUBLIC
    CUDA::cudart
    ${TRT_LIBS}
    ${OpenCV_LIBS}
    core_common
    core_interfaces
    core_cv
    core_algorithms
    core_trt_backend
    core_infer
    yaml-cpp
)

# ----------------------------------------
# Executable target: Object Detector
# ----------------------------------------
add_executable(sort_tracker app/track.cpp)

target_link_libraries(sort_tracker PUBLIC
    CUDA::cudart
    ${TRT_LIBS}
    ${OpenCV_LIBS}
    core_common
    core_interfaces
    core_cv
    core_algorithms
    core_trt_backend
    core_infer
    core_tracker
    yaml-cpp
)

target_include_directories(sort_tracker PUBLIC ${OpenCV_INCLUDE_DIRS} ${TRT_INCLUDES})

# ----------------------------------------
# Install
# ----------------------------------------
include(GNUInstallDirs)

install(TARGETS yolo sort_tracker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/app/config
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR})
