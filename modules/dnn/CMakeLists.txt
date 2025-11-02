# Base interfaces
add_library(base_dnn INTERFACE)
target_include_directories(base_dnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Options to enable/disable backends
option(ENABLE_CUDA "Enable CUDA backend" ON)

# Collect backends
set(DNN_BACKENDS "")

if (ENABLE_TRT)
    file(GLOB SRC "src/*.cpp")

    add_library(libtrt_backend STATIC
        ${SRC}
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
            liballoc
            libutils
    )

    list(APPEND DNN_BACKENDS libtrt_backend)
endif()

add_library(forge_dnn INTERFACE)
target_include_directories(forge_dnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(forge_dnn INTERFACE ${DNN_BACKENDS} base_dnn)

