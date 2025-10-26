add_library(core_interfaces INTERFACE)

target_include_directories(core_interfaces INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(core_interfaces INTERFACE
    core_common
    core_cv
    core_trt_backend
)
