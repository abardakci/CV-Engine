file(GLOB SRC "src/*.cpp")

add_library(libvision_toolkit STATIC
    ${SRC}
)

find_package(OpenCV REQUIRED)

target_include_directories(libvision_toolkit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(libvision_toolkit         
    PUBLIC
        libutils
    PRIVATE
        ${OpenCV_LIBS}
)
