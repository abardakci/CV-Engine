add_library(core_cv STATIC
    kf_wrapper.cpp
    kf_wrapper.hpp
    drawer.cpp
    drawer.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(core_cv 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_cv 
    PUBLIC
        ${OpenCV_LIBS}
        core_common
        core_interfaces
)
