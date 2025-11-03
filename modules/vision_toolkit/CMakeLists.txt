file(GLOB SRC "src/*.cpp" "src/*.hpp")

add_library(libvision_toolkit STATIC
    ${SRC}
)

find_package(OpenCV REQUIRED)

target_include_directories(libvision_toolkit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(libvision_toolkit         
    PUBLIC
        libutils
    PRIVATE
        ${OpenCV_LIBS}
)
