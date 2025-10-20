add_library(core_algorithms STATIC
    hungarian.cpp
    hungarian.hpp
    postprocess.cpp
    postprocess.hpp
)

target_include_directories(core_algorithms PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(core_algorithms PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_algorithms PRIVATE
    ${OpenCV_LIBS}
    core_common
)
