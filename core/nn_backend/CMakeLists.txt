add_library(core_engine STATIC
    trt_engine.cpp
)

target_include_directories(core_engine PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(core_engine PUBLIC
    core_common
)
