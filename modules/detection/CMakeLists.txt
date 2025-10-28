add_library(libdetection STATIC
    yolo_inference.cpp
    yolo_inference.hpp
    yolo_cfg.cpp
    yolo_cfg.hpp
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(libdetection 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)   

target_link_libraries(libdetection
    PUBLIC
        ${OpenCV_LIBS}
        core_utils
        core_interfaces
        libvision_toolkit

    PRIVATE
        yaml-cpp
)
