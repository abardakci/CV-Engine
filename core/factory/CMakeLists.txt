add_library(core_factory STATIC
    engine_factory.cpp
    kalman_factory.cpp
    engine_factory.hpp
    kalman_factory.hpp
)

target_include_directories(core_factory 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_factory
    PUBLIC
        core_common
        core_interfaces

    PRIVATE
        core_trt_backend
        core_cv
)
