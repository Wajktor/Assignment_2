#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define threadBlocks 1
#define threads 256

__global__ void printKernel()
{
	printf("Hello World!  My threadId is %d \n", threadIdx.x);
}

int main()
{
	cudaDeviceSynchronize();	
	printKernel <<< threadBlocks, threads >>>();

	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
	} else {
		printf("done");
	}
	return 0;
}