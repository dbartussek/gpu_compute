#include <stdint.h>

#include <cuda_runtime.h>

__global__ void kernelSumReduce(uint32_t *g_idata, uint32_t *g_odata) {
    __shared__ uint32_t sdata[256];

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = g_idata[i];

    __syncthreads();
    for (size_t s=1; s < blockDim.x; s *=2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(g_odata, sdata[0]);
    }
}

__global__ void
kernelSum(const uint32_t* data, uint32_t* output, size_t work_count, size_t threads) {
    size_t my_id = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t acc = 0;

    for (size_t i = 0; i < work_count; i++) {
        // acc += data[my_id * work_count + i];
        acc += data[my_id + threads * i];
    }

    output[my_id] = acc;
}

__global__ void
kernelSumFinal(uint32_t* data, size_t size) {
    for (size_t i = 1; i < size; i++) {
        data[0] += data[i];
    }
}
__global__ void
kernelEmpty() {
}

uint32_t* U32_DATA = nullptr;
size_t U32_DATA_SIZE = 0;

uint32_t* U32_OUTPUT_BUFFER = nullptr;
uint32_t* U32_OUTPUT_BUFFER_CPU = nullptr;
size_t U32_OUTPUT_BUFFER_SIZE = 0;


extern  "C" {
    void cuda_empty_kernel() {
        kernelEmpty<<<1, 1>>>();
        cudaDeviceSynchronize();
    }

    void cuda_accumulate_u32_free_data() {
        if (U32_DATA != nullptr) {
            cudaFree(U32_DATA);
            U32_DATA = nullptr;
            U32_DATA_SIZE = 0;
        }
    }
    void cuda_accumulate_u32_set_data(uint32_t* data, size_t count) {
        cuda_accumulate_u32_free_data();

        cudaMalloc(&U32_DATA, sizeof(uint32_t) * count);
        U32_DATA_SIZE = count;

        cudaMemcpy(U32_DATA, data, sizeof(uint32_t) * count, cudaMemcpyHostToDevice);
    }


    uint32_t cuda_accumulate_u32_sum_subgroup() {
        uint32_t* output;
        cudaMalloc(&output, sizeof(uint32_t));
        cudaMemset(output, 0, sizeof(uint32_t));

        size_t subgroup_size = min(U32_DATA_SIZE, (size_t) 256);

        kernelSumReduce<<<U32_DATA_SIZE/subgroup_size, subgroup_size>>>(U32_DATA, output);
        cudaDeviceSynchronize();

        uint32_t result;
        cudaMemcpy(&result, output, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        cudaFree(output);
        return result;
    }

    uint32_t cuda_accumulate_u32_sum(
            size_t total_threads,
            size_t subgroup_size,

            size_t second_accumulate_on_gpu
    ) {
        if (U32_OUTPUT_BUFFER_SIZE != total_threads) {
            if (U32_OUTPUT_BUFFER) {
                cudaFree(U32_OUTPUT_BUFFER);
                free(U32_OUTPUT_BUFFER_CPU);
            }

            cudaMalloc(&U32_OUTPUT_BUFFER, sizeof(uint32_t) * total_threads);
            U32_OUTPUT_BUFFER_CPU = (uint32_t*) malloc(sizeof(uint32_t) * total_threads);
            U32_OUTPUT_BUFFER_SIZE = total_threads;
        }

        kernelSum<<<total_threads / subgroup_size, subgroup_size>>>(U32_DATA, U32_OUTPUT_BUFFER, U32_DATA_SIZE / total_threads, total_threads);
        if (second_accumulate_on_gpu) {
            kernelSumFinal<<<1, 1>>>(U32_OUTPUT_BUFFER, total_threads);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(
                U32_OUTPUT_BUFFER_CPU,
                U32_OUTPUT_BUFFER,
                sizeof(uint32_t) * (second_accumulate_on_gpu ? second_accumulate_on_gpu : total_threads),
                cudaMemcpyDeviceToHost
        );

        if (second_accumulate_on_gpu != 1) {
            if (second_accumulate_on_gpu) {
                for (size_t i = 1; i < second_accumulate_on_gpu; i++) {
                    U32_OUTPUT_BUFFER_CPU[0] += U32_OUTPUT_BUFFER_CPU[i];
                }
            } else {
                for (size_t i = 1; i < total_threads; i++) {
                    U32_OUTPUT_BUFFER_CPU[0] += U32_OUTPUT_BUFFER_CPU[i];
                }
            }
        }
        uint32_t result = U32_OUTPUT_BUFFER_CPU[0];

        return result;
    }

}
