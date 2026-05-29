#include <cuda_runtime.h>
#include <stddef.h>

extern "C" {

void* cuda_allocate_vram(size_t size_bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size_bytes);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}

void cuda_upload(const void* host_ptr, void* device_ptr, size_t size_bytes) {
    // cudaMemcpyHostToDevice
    cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
}

void cuda_download(const void* device_ptr, void* host_ptr, size_t size_bytes) {
    // cudaMemcpyDeviceToHost
    cudaMemcpy(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost);
}

void cuda_free_vram(void* device_ptr) {
    if (device_ptr != nullptr) {
        cudaFree(device_ptr);
    }
}

} // extern "C"