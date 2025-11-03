### lib: allocator_interface
add_library(allocator_interface INTERFACE)
target_sources(allocator_interface
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/base_allocator.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_factory.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/allocator_interface.hpp
)

### Collect backend libs
set(ALLOCATOR_BACKENDS "")

### lib: liballoc_cuda
option(ENABLE_CUDA "Enable CUDA backend" ON)
if (ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(liballoc_cuda STATIC
        "src/cuda_allocator.cpp"
        "src/cuda_allocator.hpp"
        "src/cuda_utils.hpp"
    )

    target_link_libraries(liballoc_cuda 
        PUBLIC
            CUDA::cudart
            allocator_interface
    )

    list(APPEND ALLOCATOR_BACKENDS liballoc_cuda)
endif()

### lib: liballoc
add_library(liballoc STATIC 
    "src/base_allocator.hpp"
    "src/allocator_interface.hpp"
    "src/allocator_factory.cpp"    
    "src/allocator_factory.hpp"    
)

target_link_libraries(liballoc
    PUBLIC
        allocator_interface    
    PRIVATE
        ${ALLOCATOR_BACKENDS}  # backends (liballoc_cuda, liballoc_rocm, ...)
)

target_include_directories(liballoc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
