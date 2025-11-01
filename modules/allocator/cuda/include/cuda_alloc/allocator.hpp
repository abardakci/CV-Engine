#include "../../include/base_allocator/allocator.hpp"
#include <cuda_runtime_api.h>
#include "utils.hpp"

#include <iostream>

template <>
struct backend::Traits<backend::Tag::CUDA>
{
    using StreamType = cudaStream_t;
    static void malloc(void **ptr, size_t bytes) { CUDA_CHECK(cudaMalloc(ptr, bytes)); }
    static void free(void *ptr) { CUDA_CHECK(cudaFree(ptr)); }
    static void memcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind copy_mode) { CUDA_CHECK(cudaMemcpy(dst, src, bytes, copy_mode)); }
    static void memcpy_async(void *dst, const void *src, size_t bytes, cudaMemcpyKind copy_mode, cudaStream_t stream) { CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, copy_mode, stream)); }
};


class CudaAllocator : public BaseAllocator<CudaAllocator, backend::Tag::CUDA>
{
public:
    void allocate_impl(void **ptr, size_t bytes);
    void free_impl(void *ptr);
    void memcpy_impl(void *dst, void *src, size_t bytes, CopyMode copy_mode);
    void memcpy_async_impl(void *dst, void *src, size_t bytes, CopyMode copy_mode, void *stream);

private:
    inline static cudaMemcpyKind to_cuda_copy_mode(CopyMode copy_mode)
    {
        switch (copy_mode)
        {
            case CopyMode::DeviceToDevice: return cudaMemcpyDeviceToDevice;
            case CopyMode::DeviceToHost:   return cudaMemcpyDeviceToHost;
            case CopyMode::HostToDevice:   return cudaMemcpyHostToDevice;
            case CopyMode::HostToHost:     return cudaMemcpyHostToHost;
            default: std::cerr << "[ERROR] to_cuda_copy_mode returned None \n"; 
        }
        
        return cudaMemcpyDeviceToDevice;
    }
};
