add_library(libtrackers STATIC
    sort_tracker.cpp
    sort_tracker.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(libtrackers 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)       

target_link_libraries(libtrackers
    PUBLIC
        core_utils
        core_interfaces
        libvision_toolkit
        libfactory
    PRIVATE
        ${OpenCV_LIBS}
)
