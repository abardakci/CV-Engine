add_library(liballoc_cuda STATIC
    src/allocator.cpp
)

find_package(CUDAToolkit REQUIRED)

### API
target_include_directories(liballoc_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(liballoc_cuda 
    PUBLIC
        CUDA::cudart
)
