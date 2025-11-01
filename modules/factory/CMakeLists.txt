add_library(libfactory STATIC
    kalman_factory.cpp
    kalman_factory.hpp
)

target_include_directories(libfactory 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(libfactory
    PUBLIC
        core_utils
        core_interfaces

    PRIVATE
        libtrt_backend
        libkalman_filter
)
