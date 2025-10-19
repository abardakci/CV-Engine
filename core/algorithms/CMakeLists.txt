add_library(core_algo STATIC
    hungarian.cpp
    postprocess.cpp
)

target_include_directories(core_algo PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_algo PUBLIC
    ${OpenCV_LIBS}
    core_common
)
