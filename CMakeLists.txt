cmake_minimum_required(VERSION 3.18)

project(VisionEngine LANGUAGES CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB SRC "src/*.cpp")

add_executable(SortTracker ${SRC})

target_link_libraries(SortTracker PRIVATE
    ${TRT_LIBS}
    ${OpenCV_LIBS}
    CUDA::cudart
)

target_include_directories(SortTracker PRIVATE
    ${TRT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

# For windows
# # DLL copy to exe dir
# file(GLOB TENSORRT_DLLS "${TENSORRT_ROOT}/lib/*.dll")

# add_custom_command(TARGET VisionEngine POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         ${TENSORRT_DLLS}
#         $<TARGET_FILE_DIR:VisionEngine>
# )
