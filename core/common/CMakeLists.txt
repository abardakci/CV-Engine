add_library(core_common STATIC
    types.cpp
    types.hpp
    file_io.cpp
    file_io.hpp
    timer.cpp
    timer.hpp
)

target_include_directories(core_common PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
