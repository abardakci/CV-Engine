add_library(core_infer STATIC
    yolo_detector.cpp
    yolo_detector.hpp
    esrgan_upscaler.cpp
    esrgan_upscaler.hpp
)

target_include_directories(core_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(core_infer PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_infer PRIVATE
    ${OpenCV_LIBS}
    core_common
    core_algorithms
    core_interfaces
)
