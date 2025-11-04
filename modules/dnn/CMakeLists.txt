add_library(dnn_interface INTERFACE)
target_include_directories(dnn_interface 
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)



add_library(dnn_common STATIC 
    "src/buffer.cpp"
    "src/buffer.hpp"
    "src/engine_factory.hpp"
    "src/tensor.cpp"
    "src/tensor.hpp"
    "src/tensor_factory.hpp"
)

target_link_libraries(dnn_common 
    PUBLIC
        dnn_interface
        libutils
        liballoc
)

target_include_directories(dnn_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
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
        "src/engine_factory.cpp"
        "src/tensor_factory.cpp"
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
            dnn_common
    )

    list(APPEND DNN_BACKENDS libtrt)
endif()



add_library(libdnn STATIC)

# PUBLIC include path
target_include_directories(libdnn
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# PUBLIC: herkes sees dnn_interface + common API
target_link_libraries(libdnn
    PUBLIC
        dnn_interface
        dnn_common
    PRIVATE
        ${DNN_BACKENDS}      # ✅ backend'ler içte saklanır
)

