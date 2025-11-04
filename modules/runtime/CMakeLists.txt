file(GLOB SRC "src/*.cpp" "src/*.hpp")

add_library(libruntime STATIC
    ${SRC}
)

find_package(OpenCV REQUIRED)

target_include_directories(libruntime 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(libruntime
    PRIVATE
        ${OpenCV_LIBS}
        libvision
        libutils
        libvision_toolkit
        libdnn
)
