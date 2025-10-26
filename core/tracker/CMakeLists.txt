add_library(core_tracker STATIC
    sort_tracker.cpp
)

target_include_directories(core_tracker PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(core_tracker PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_tracker PRIVATE
    ${OpenCV_LIBS}
    core_common
    core_cv
    core_algorithms
    core_interfaces
    core_factory
)
