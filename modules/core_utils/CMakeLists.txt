add_library(core_utils STATIC
    types.cpp
    types.hpp
    file_io.cpp
    file_io.hpp
    timer.hpp
)

target_include_directories(core_utils 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_utils
    PUBLIC 
        core_interfaces
)
