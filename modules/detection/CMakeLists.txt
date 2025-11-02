file(GLOB SRC "src/*.cpp")

add_library(libdetection STATIC
    ${SRC}
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(libdetection 
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)   

target_link_libraries(libdetection
    PUBLIC
        ${OpenCV_LIBS}
        libutils
        libvision_toolkit

    PRIVATE
        yaml-cpp
)
