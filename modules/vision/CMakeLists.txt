file(GLOB SRC 
    "src/yolo/**.cpp" 
    "src/yolo/**.hpp" 
    "src/upscaler/**.cpp"
    "src/upscaler/**.hpp")

add_library(libvision STATIC
    ${SRC}
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

target_include_directories(libvision
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(libvision
    PUBLIC
        ${OpenCV_LIBS}
        libutils
        libvision_toolkit
        dnn_interface

    PRIVATE
        yaml-cpp
)
