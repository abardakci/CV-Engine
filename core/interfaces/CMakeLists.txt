add_library(core_interfaces INTERFACE)

target_include_directories(core_interfaces INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_interfaces INTERFACE
    core_common
)
