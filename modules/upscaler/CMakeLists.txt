add_library(libupscaler STATIC
    esrgan_inference.cpp
    esrgan_inference.hpp
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(libupscaler
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)   

target_link_libraries(libupscaler
    PUBLIC
        ${OpenCV_LIBS}
        core_utils
        core_interfaces
        libvision_toolkit

    PRIVATE
        yaml-cpp
)
