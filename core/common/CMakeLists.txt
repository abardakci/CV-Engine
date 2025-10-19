add_library(core_common STATIC
    types.cpp
    file_io.cpp
    timer.cpp
)

target_include_directories(core_common PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
