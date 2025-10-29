add_library(libtrt_backend STATIC
    engine.cpp
    engine.hpp
    cuda_allocator.cpp
    cuda_allocator.hpp
    cuda_buffer.cpp
    cuda_buffer.hpp
    tensor.cpp
    tensor.hpp
    tensor_factory.cpp
    tensor_factory.hpp
    helpers.cpp
    helpers.hpp
    logger.hpp
)

find_package(CUDAToolkit REQUIRED)
set(TRT_INCLUDES /usr/include/x86_64-linux-gnu)
set(TRT_LIBS
    /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.10.0
    /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10.10.0
)

target_include_directories(libtrt_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TRT_INCLUDES}
)

target_link_libraries(libtrt_backend 
    PUBLIC
        CUDA::cudart
        ${TRT_LIBS}
        core_utils
        core_interfaces
)
