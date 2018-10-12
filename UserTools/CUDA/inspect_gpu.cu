//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <helper_cuda.h>
#include <sys/time.h>

// CUDA = Computer Device Unified Architecture

void print_gpu_properties();

//
// main code
//

int main(int argc, const char **argv)
{


  /////////////////////
  // initialise card //
  /////////////////////
  findCudaDevice(argc, argv);


  ////////////////////
  // inspect device //
  ////////////////////
  print_gpu_properties();


  return 1;
}


// Print device properties
void print_gpu_properties(){

  int devCount;
  cudaGetDeviceCount(&devCount);
  printf(" CUDA Device Query...\n");
  printf(" There are %d CUDA devices.\n", devCount);
  cudaDeviceProp devProp;
  for (int i = 0; i < devCount; ++i){
    // Get device properties
    printf(" CUDA Device #%d\n", i);
    cudaGetDeviceProperties(&devProp, i);
    printf("Major revision number:          %d\n",  devProp.major);
    printf("Minor revision number:          %d\n",  devProp.minor);
    printf("Name:                           %s\n",  devProp.name);
    printf("Total global memory:            %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block:  %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:      %d\n",  devProp.regsPerBlock);
    printf("Warp size:                      %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:           %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:      %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of block:   %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
      printf("Maximum dimension %d of grid:    %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                     %d\n",  devProp.clockRate);
    printf("Total constant memory:          %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:              %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution:  %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:      %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:       %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    printf("Memory Clock Rate (KHz):        %d\n", devProp.memoryClockRate);
    printf("Memory Bus Width (bits):        %d\n", devProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s):   %f\n", 2.0*devProp.memoryClockRate*(devProp.memoryBusWidth/8)/1.0e6);
    printf("Concurrent kernels:             %d\n",  devProp.concurrentKernels);
  }
  size_t available_memory, total_memory;
  cudaMemGetInfo(&available_memory, &total_memory);
  size_t stack_memory;
  cudaDeviceGetLimit(&stack_memory, cudaLimitStackSize);
  size_t fifo_memory;
  cudaDeviceGetLimit(&fifo_memory, cudaLimitPrintfFifoSize);
  size_t heap_memory;
  cudaDeviceGetLimit(&heap_memory, cudaLimitMallocHeapSize);
  printf(" memgetinfo: available_memory %f MB, total_memory %f MB, stack_memory %f MB, fifo_memory %f MB, heap_memory %f MB \n", (double)available_memory/1.e6, (double)total_memory/1.e6, (double)stack_memory/1.e6, (double)fifo_memory/1.e6, (double)heap_memory/1.e6);


  return;
}
