#include "cuda_runtime.h"
#include <cassert>
#include <random>
#include <algorithm>
#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define PXL_HOST_LOOPS 64

template <class T1>
__global__ void gather_kernel(const unsigned int *const __restrict__ position,
                              const T1 *const __restrict__ in1, T1 *out1,
                              const size_t n) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    out1[i] = in1[position[i]];
  }
}

int main(int argc, char **argv) {
  const size_t size_MB = max(atoi(argv[1]), 1) * sizeof(unsigned int);
  const size_t size = size_MB * 1024 * 1024;

  thrust::host_vector<unsigned int> index(size);
  thrust::sequence(index.begin(), index.end());

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(index.begin(), index.end(), g);

  thrust::device_vector<unsigned int> index_d = index;
  thrust::device_vector<unsigned int> data_d = index;
  thrust::device_vector<unsigned int> out_d = index;

  int minGridSize;
  int blockSize;
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     gather_kernel<unsigned int>, 0, 0);

  int gridSize = (size + blockSize - 1) / blockSize;

  // warm-up
  gather_kernel<<<gridSize, blockSize>>>(
      thrust::raw_pointer_cast(index_d.data()),
      thrust::raw_pointer_cast(data_d.data()),
      thrust::raw_pointer_cast(out_d.data()), size);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord(start);

  for (int ii = 0; ii < PXL_HOST_LOOPS; ii++) {
    gather_kernel<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(index_d.data()),
        thrust::raw_pointer_cast(data_d.data()),
        thrust::raw_pointer_cast(out_d.data()), size);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, end);

  printf("%8lu, %f, %8.4f\n", size_MB, elapsed / PXL_HOST_LOOPS,
         (1.0 * size_MB * PXL_HOST_LOOPS) / (elapsed * 1024));

  return 0;
}
