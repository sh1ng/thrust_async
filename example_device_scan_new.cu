/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Simple example of DeviceScan::ExclusiveSum().
 *
 * Computes an exclusive sum of int keys.
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_device_scan.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/


#include <stdio.h>

#include <cub/util_allocator.cuh>
#include <cub/device/device_scan.cuh>


using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


//---------------------------------------------------------------------
// Test generation
//---------------------------------------------------------------------


/**
 * Initialize problem
 */
void Initialize(
    int        *h_in,
    int          num_items)
{
    for (int i = 0; i < num_items; ++i)
        h_in[i] = i;

}



//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items = 1 << 28;


    printf("cub::DeviceScan::ExclusiveSum %d items (%d-byte elements)\n",
        num_items, (int) sizeof(int));
    fflush(stdout);

    // Allocate host arrays
    int*  h_in_1 = new int[num_items];
    int*  h_reference_1 = new int[num_items];
    int*  h_in_2 = new int[num_items];
    int*  h_reference_2 = new int[num_items];

    cudaStream_t s1;
    cudaStream_t s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);



    // Initialize problem and solution
    Initialize(h_in_1, num_items);

    Initialize(h_in_1, num_items);

    // Allocate problem device arrays
    int *d_in_1 = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in_1, sizeof(int) * num_items, s1));

    int *d_in_2 = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_in_2, sizeof(int) * num_items, s2));


    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in_1, h_in_1, sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in_2, h_in_2, sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // Allocate device output array
    int *d_out_1 = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_1, sizeof(int) * num_items, s1));

    // Allocate device output array
    int *d_out_2 = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)&d_out_2, sizeof(int) * num_items, s2));


    // Allocate temporary storage
    void            *d_temp_storage_1 = NULL;
    size_t          temp_storage_bytes_1 = 0;
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage_1, temp_storage_bytes_1, d_in_1, d_out_1, num_items, s1));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_1, temp_storage_bytes_1, s1));

    // Allocate temporary storage
    void            *d_temp_storage_2 = NULL;
    size_t          temp_storage_bytes_2 = 0;
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage_2, temp_storage_bytes_2, d_in_2, d_out_2, num_items, s2));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage_2, temp_storage_bytes_2, s2));


    // Run
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage_1, temp_storage_bytes_1, d_in_1, d_out_1, num_items, s1));

    // Run
    CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage_2, temp_storage_bytes_2, d_in_2, d_out_2, num_items, s2));

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);



    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);


    return 0;
}



