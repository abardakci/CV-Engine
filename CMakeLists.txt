cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES C CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT kök dizinini dışarıdan alınabilir hale getir
set(TENSORRT_ROOT "C:/TensorRT-10.9.0.34" CACHE PATH "Path to TensorRT root directory")

# TensorRT lib dosyaları
set(TENSORRT_LIBS
    ${TENSORRT_ROOT}/lib/nvinfer_10.lib
    ${TENSORRT_ROOT}/lib/nvinfer_dispatch_10.lib
    ${TENSORRT_ROOT}/lib/nvinfer_lean_10.lib
    ${TENSORRT_ROOT}/lib/nvinfer_plugin_10.lib
    ${TENSORRT_ROOT}/lib/nvinfer_vc_plugin_10.lib
    ${TENSORRT_ROOT}/lib/nvonnxparser_10.lib
)

# Proje kaynak dosyalarını al
file(GLOB SRC
    "src/*.cpp"
    "src/*.c"
    "src/*.hpp"
)

# Executable oluştur
add_executable(VisionEngine ${SRC})

# Bağlantı ayarları
target_link_libraries(VisionEngine PRIVATE
    ${TENSORRT_LIBS}
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Include path ayarları
target_include_directories(VisionEngine PRIVATE
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

# DLL dosyalarını build klasörüne kopyala
file(GLOB TENSORRT_DLLS "${TENSORRT_ROOT}/lib/*.dll")

add_custom_command(TARGET VisionEngine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TENSORRT_DLLS}
        $<TARGET_FILE_DIR:VisionEngine>
)
