add_library(core_trt_backend STATIC
    trt_engine.cpp
    trt_engine.hpp
)

target_include_directories(core_trt_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TRT_INCLUDES}
)

target_link_libraries(core_trt_backend PRIVATE
    core_common
    core_interfaces
    CUDA::cudart
    ${TRT_LIBS}
)
