add_library(libkalman_filter STATIC
    kf_wrapper.cpp
    kf_wrapper.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(libkalman_filter 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(libkalman_filter 
    PUBLIC
        core_utils
        core_interfaces
        ${OpenCV_LIBS}
)
