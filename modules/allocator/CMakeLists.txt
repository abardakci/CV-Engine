### Header-only base interface
add_library(base_allocator INTERFACE)
target_include_directories(base_allocator INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

### Collect backend libs
set(ALLOCATOR_BACKENDS "")
set(ALLOCATOR_TESTS "")

### Option to enable CUDA backend
option(ENABLE_CUDA "Enable CUDA backend" ON)
if (ENABLE_CUDA)
    add_subdirectory(cuda)
    file(GLOB cuda_tests "cuda/test/*.cpp")

    list(APPEND ALLOCATOR_BACKENDS liballoc_cuda)
    list(APPEND ALLOCATOR_TESTS ${cuda_tests})
endif()

### liballoc static lib
add_library(liballoc STATIC
    src/factory.cpp
)

target_include_directories(liballoc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(liballoc
    PRIVATE
        ${ALLOCATOR_BACKENDS}     # backends (liballoc_cuda, liballoc_rocm, ...)
    PUBLIC
        base_allocator            # interface header-only dependency
)

### add test executables
foreach(test_src ${TESTS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE liballoc)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
