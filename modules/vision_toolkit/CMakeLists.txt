add_library(libvision_toolkit STATIC
    hungarian.cpp
    hungarian.hpp
    preproc.cpp
    preproc.hpp
    postproc.cpp
    postproc.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(libvision_toolkit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(libvision_toolkit 
    PUBLIC
        core_utils
        
    PRIVATE
        ${OpenCV_LIBS}
)
