cmake_minimum_required(VERSION 3.18)

project(RealUpscale LANGUAGES C CXX)

# CUDA Toolkit bulunuyor
find_package(CUDAToolkit REQUIRED)

# OpenCV bulunuyor
find_package(OpenCV REQUIRED)

# TensorRT kök dizini
set(TENSORRT_ROOT "C:/TensorRT-10.9.0.34")

# CUDA compute capability: 8.6 → Ampere kartları (örneğin RTX 30 serisi)
# set(CMAKE_CUDA_ARCHITECTURES 86)
# set(CMAKE_CUDA_STANDARD 14)

# TensorRT include klasörü
# include_directories(
#     ${TENSORRT_ROOT}/include
#     ${OpenCV_INCLUDE_DIRS}
# )

# TensorRT lib dosyaları - **MANUEL** belirt
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
add_executable(RealUpscale ${SRC})

# Bağlantı ayarları
target_link_libraries(RealUpscale PRIVATE
    ${TENSORRT_LIBS}
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Eğer istersen include path'leri target'a bağlayarak da yapabilirsin (daha modern)
target_include_directories(RealUpscale PRIVATE
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

# find dll folder and copy dlls
file(GLOB TENSORRT_DLLS "C:/TensorRT-10.9.0.34/lib/*.dll")

add_custom_command(TARGET RealUpscale POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TENSORRT_DLLS}
        $<TARGET_FILE_DIR:RealUpscale>
)