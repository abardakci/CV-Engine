add_library(core_cv STATIC
    opencv_kalman.cpp
    opencv_kalman.hpp
    drawer.cpp
    drawer.hpp
)

target_include_directories(core_cv PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(core_cv PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_cv PRIVATE
    ${OpenCV_LIBS}
    core_common
    core_interfaces
)
