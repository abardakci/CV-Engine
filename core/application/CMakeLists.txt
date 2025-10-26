add_library(core_app STATIC
    yolo.cpp
    yolo.hpp
)

find_package(OpenCV REQUIRED)

target_include_directories(core_app 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_app
    PRIVATE
        ${OpenCV_LIBS}
        core_infer
        core_factory
        core_cv
)
