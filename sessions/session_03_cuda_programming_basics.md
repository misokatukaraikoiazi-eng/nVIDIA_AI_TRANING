# Session 3: CUDA Programming Basics

## Learning Objectives

By the end of this session you will be able to:

- Explain the CUDA programming model (threads, blocks, grids)
- Write and compile a basic CUDA kernel in C++
- Understand memory hierarchy and common memory-access patterns
- Profile a CUDA application with Nsight Systems

---

## 1. What Is CUDA?

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It lets developers write C/C++ (and other language) code that runs directly on the GPU.

Key terms:

| Term | Meaning |
|------|---------|
| **Host** | CPU and its memory |
| **Device** | GPU and its memory |
| **Kernel** | A function that runs on the GPU, called from the CPU |
| **Thread** | The smallest unit of execution on the GPU |
| **Block** | A group of threads that share fast shared memory |
| **Grid** | A group of blocks; one grid per kernel launch |

---

## 2. Thread Hierarchy

```
Grid
 └─ Block (0,0)  Block (1,0)  Block (2,0) ...
      └─ Thread(0,0)  Thread(1,0)  Thread(2,0) ...
```

Inside a kernel, each thread identifies itself using built-in variables:

```cpp
int block_id  = blockIdx.x;
int thread_id = threadIdx.x;
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 3. Your First CUDA Kernel

```cpp
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 20;  // 1 M elements
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialise inputs
    for (int i = 0; i < N; i++) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy host → device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // Copy device → host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Verify
    for (int i = 0; i < N; i++) {
        if (h_c[i] != 3.0f) { printf("Error at %d\n", i); return 1; }
    }
    printf("All results correct!\n");

    // Free memory
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
```

### Compile and Run

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

---

## 4. Memory Hierarchy

| Memory Type | Location | Scope | Latency | Size |
|------------|----------|-------|---------|------|
| Registers | On-chip | Per-thread | ~1 cycle | Small (~256 KB / SM) |
| Shared memory | On-chip | Per-block | ~32 cycles | 48–228 KB / SM |
| L2 cache | On-chip | Device-wide | ~200 cycles | 40–80 MB |
| Global memory | Off-chip (HBM) | Device-wide | ~600 cycles | Up to 80 GB (H100) |
| Constant memory | Off-chip (cached) | Device-wide | ~1 cycle (hit) | 64 KB |

**Key tip:** Coalesced global memory accesses (consecutive threads reading consecutive addresses) are critical for performance.

---

## 5. Using Shared Memory

```cpp
__global__ void tiled_matrix_mul(const float* A, const float* B, float* C, int N) {
    __shared__ float tile_A[16][16];
    __shared__ float tile_B[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / 16; t++) {
        tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * 16 + threadIdx.x];
        tile_B[threadIdx.y][threadIdx.x] = B[(t * 16 + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < 16; k++)
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
```

`__syncthreads()` ensures all threads in a block have loaded the tile before computation begins.

---

## 6. Profiling with Nsight Systems

```bash
# Profile an application
nsys profile --stats=true ./vector_add

# Open the generated report in the Nsight Systems GUI
nsys-ui report1.nsys-rep
```

Key metrics to watch:

- **SM utilisation** – Are all cores busy?
- **Memory bandwidth** – Are you saturating HBM bandwidth?
- **Kernel duration** – How long does each kernel run?
- **Memory transfers** – Are host↔device copies a bottleneck?

---

## 7. Common Pitfalls

| Pitfall | Symptom | Fix |
|--------|---------|-----|
| Uncoalesced memory access | Low memory bandwidth | Ensure threads access sequential addresses |
| Warp divergence | Low SM utilisation | Minimise if/else inside kernels |
| Missing `__syncthreads()` | Race conditions, wrong results | Add sync after shared memory writes |
| Memory leak | `cudaErrorMemoryAllocation` | Always `cudaFree` after use |
| Ignoring CUDA errors | Silent failures | Check return codes with `cudaGetLastError()` |

---

## Hands-On Exercise

1. Compile and run the `vector_add.cu` example above.
2. Modify the kernel to compute element-wise multiplication instead of addition.
3. Profile with `nsys profile` and identify how long the kernel takes vs. the memory transfers.

---

## Key Takeaways

- CUDA exposes massive GPU parallelism through a thread/block/grid hierarchy.
- Shared memory is a programmer-managed cache that can dramatically reduce global memory bandwidth pressure.
- Profiling early and often is essential for writing performant CUDA code.

---

## Additional Resources

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [CUDA Samples on GitHub](https://github.com/NVIDIA/cuda-samples)

---

**Previous:** [Session 2 – Deep Learning Fundamentals](session_02_deep_learning_fundamentals.md)  
**Next:** [Session 4 – NVIDIA AI Frameworks](session_04_nvidia_ai_frameworks.md)
