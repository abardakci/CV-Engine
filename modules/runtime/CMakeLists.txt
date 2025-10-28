add_library(libruntime STATIC
    draw.cpp
    draw.hpp
    yolo_runner.cpp
    yolo_runner.hpp
    tracker_runner.cpp
    tracker_runner.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(libruntime 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(libruntime
    PRIVATE
        ${OpenCV_LIBS}
        libdetection
        libfactory
        libkalman_filter
        libtrackers
        libupscaler
)
