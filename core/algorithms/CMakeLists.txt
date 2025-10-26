add_library(core_algorithms STATIC
    hungarian.cpp
    hungarian.hpp
    postprocess.cpp
    postprocess.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(core_algorithms
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_algorithms 
    PUBLIC
        ${OpenCV_LIBS}
        core_common
)
