file(GLOB SRC "src/*.cpp")

add_library(libtrackers STATIC
    ${SRC}
)

find_package(OpenCV REQUIRED)

target_include_directories(libtrackers 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)       

target_link_libraries(libtrackers
    PUBLIC
        libutils
        libvision_toolkit
    PRIVATE
        ${OpenCV_LIBS}
        yaml-cpp
)
