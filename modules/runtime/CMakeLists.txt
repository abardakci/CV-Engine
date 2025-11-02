file(GLOB SRC "src/*.cpp")

add_library(libruntime STATIC
    ${SRC}
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
        libutils
        libvision_toolkit
)
