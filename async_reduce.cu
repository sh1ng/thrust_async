#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cassert>

#if __cplusplus >= 201103L
#include <future>
#endif

// This example demonstrates two ways to achieve algorithm invocations that are asynchronous with
// the calling thread.
//
// The first method wraps a call to thrust::reduce inside a __global__ function. Since __global__ function
// launches are asynchronous with the launching thread, this achieves asynchrony. The result of the reduction
// is stored to a pointer to CUDA global memory. The calling thread waits for the result of the reduction to 
// be ready by synchronizing with the CUDA stream on which the __global__ function is launched.
//
// The second method uses the C++11 library function, std::async, to create concurrency. The lambda function
// given to std::async returns the result of thrust::reduce to a std::future. The calling thread can use the
// std::future to wait for the result of the reduction. This method requires a compiler which supports
// C++11-capable language and library constructs.

template<typename Iterator, typename T, typename BinaryOperation, typename Pointer>
__global__ void reduce_kernel(Iterator first, Iterator last, T init, BinaryOperation binary_op, Pointer result)
{
  *result = thrust::reduce(thrust::cuda::par, first, last, init, binary_op);
}

int main()
{
  const size_t size = 10;
  cudaStream_t *streams = new cudaStream_t[size];
  thrust::device_vector<unsigned int> *result = new thrust::device_vector<unsigned int>[size];
  for(size_t i =0; i < size; ++i){
    result[i] = thrust::device_vector<unsigned int>(1, 0);
    cudaStream_t s;
    cudaStreamCreate(&s);
    streams[i] = s;

  }
  size_t n = 1 << 20;
  

  for(size_t i = 0; i < size; ++i){
    thrust::device_vector<unsigned int> data(n, 1);
    cudaStream_t s = streams[i];
    reduce_kernel<<<1,1,0,s>>>(data.begin(), data.end(), 0, thrust::plus<int>(), result[i].data());
  
  }

  for(size_t i =0; i < size; ++i){
    cudaStream_t s = streams[i];
    cudaStreamSynchronize(s);
  }


  // reset the result
  // result[0] = 0;

#if __cplusplus >= 201103L
  // method 2: use std::async to create asynchrony

  // copy all the algorithm parameters
  auto begin        = data.begin();
  auto end          = data.end();
  unsigned int init = 0;
  auto binary_op    = thrust::plus<unsigned int>();

  // std::async captures the algorithm parameters by value
  // use std::launch::async to ensure the creation of a new thread
  std::future<unsigned int> future_result = std::async(std::launch::async, [=]
  {
    return thrust::reduce(begin, end, init, binary_op);
  });

  // wait on the result and check that it is correct
  assert(future_result.get() == n);
#endif

  return 0;
}

