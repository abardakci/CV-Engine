add_library(dnn_interface INTERFACE)
target_sources(dnn_interface
    INTERFACE
        "src/engine_interface.hpp"
        "src/engine_factory.hpp"
)


# Collect backends
set(DNN_BACKENDS "")

option(ENABLE_TENSORRT "Enable TensorRT backend" ON)
if (ENABLE_TENSORRT)
    add_library(libtrt STATIC
        "src/trt_engine.cpp"
        "src/trt_engine.hpp"
        "src/trt_helpers.cpp"
        "src/trt_helpers.hpp"
        "src/trt_logger.hpp"
    )

    find_package(CUDAToolkit REQUIRED)

    set(TRT_INCLUDES /usr/include/x86_64-linux-gnu)
    set(TRT_LIBS
        /usr/lib/x86_64-linux-gnu/libnvinfer.so.10.10.0
        /usr/lib/x86_64-linux-gnu/libnvinfer_vc_plugin.so.10.10.0
        /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10.10.0
        /usr/lib/x86_64-linux-gnu/libnvinfer_lean.so.10.10.0
    )

    target_include_directories(libtrt
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${TRT_INCLUDES}
    )

    target_link_libraries(libtrt
        PUBLIC
            CUDA::cudart
            ${TRT_LIBS}
            liballoc
            libutils
            dnn_interface
    )

    list(APPEND DNN_BACKENDS libtrt)
endif()


add_library(libdnn STATIC 
    "src/buffer.cpp"
    "src/buffer.hpp"
    "src/engine_factory.cpp"
    "src/engine_factory.hpp"
    "src/tensor.cpp"
    "src/tensor.hpp"
    "src/tensor_factory.cpp"
    "src/tensor_factory.hpp"
)

target_link_libraries(libdnn 
    PRIVATE
        ${DNN_BACKENDS}
    PUBLIC
        dnn_interface
        libutils        
)

target_include_directories(libdnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
