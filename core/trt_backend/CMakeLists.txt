add_library(core_trt_backend STATIC
    engine.cpp
    engine.hpp
    helpers.cpp
    helpers.hpp
    logger.hpp
)

target_include_directories(core_trt_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(core_trt_backend PRIVATE
    ${TRT_INCLUDES}
)

target_link_libraries(core_trt_backend PRIVATE
    core_common
    core_interfaces
    CUDA::cudart
    ${TRT_LIBS}
)
