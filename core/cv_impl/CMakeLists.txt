add_library(core_cv STATIC
    opencv_kalman.cpp
    drawer.cpp
)

target_include_directories(core_cv PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_cv PUBLIC
    ${OpenCV_LIBS}
    core_common
)
