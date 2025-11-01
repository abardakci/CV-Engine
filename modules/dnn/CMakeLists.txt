# Base interfaces
add_library(base_dnn INTERFACE)
target_include_directories(base_dnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Options to enable/disable backends
option(ENABLE_CUDA "Enable CUDA backend" ON)

# Collect backends
set(DNN_BACKENDS "")

if (ENABLE_TRT)
    add_subdirectory(tensorrt)
    list(APPEND ALLOCATOR_BACKENDS libtrt_backend)
endif()

add_library(forge_dnn INTERFACE)
target_include_directories(forge_dnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(forge_dnn INTERFACE ${DNN_BACKENDS} base_dnn)
