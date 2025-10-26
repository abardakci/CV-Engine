add_library(core_tracker STATIC
    sort_tracker.cpp
)

target_include_directories(core_tracker 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}
)       

target_link_libraries(core_tracker
    PUBLIC
        core_common
        core_cv
        core_algorithms
        core_interfaces
        core_factory
    PRIVATE
        ${OpenCV_LIBS}
)
