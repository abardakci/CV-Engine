add_library(core_infer STATIC
    yolo_detector.cpp
    yolo_detector.hpp
    yolo_cfg.hpp
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(core_infer 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)   

target_link_libraries(core_infer 
    PUBLIC
        ${OpenCV_LIBS}
        core_common
        core_algorithms
        core_interfaces
    PRIVATE
        yaml-cpp
)
