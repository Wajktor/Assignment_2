#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include <random>
#include <iostream>
#include <math.h>


#include <stdio.h>


#define ARRAY_SIZE 10000

#define TPB 256
#define BLOCKS (ARRAY_SIZE + TPB - 1)/TPB


__global__ void saxpy(float a, float *x, float *y)
{
	int index = blockIdx.x * TPB + threadIdx.x; 
	y[index] = a * x[index] + y[index];
}


void saxpy_host(float a, float *x, float *y){
	for(int i = 0; i<ARRAY_SIZE; ++i){
		y[i] = a*x[i] + y[i];
	}
}

float compareVec(float *x, float *y){
	int length = ARRAY_SIZE;
	float sum = 0;
	for (int i = 0; i < length; ++i){
		sum += fabs(x[i] - y[i]);
	}
	return sum;
}

void random_floats(float *a, int n){

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        a[i] = dis(gen);
    }
}

int main()
{	

	const float a = 0.4;
	float *x, *y, *yCompare;

	const float d_a = a;
	float *d_x, *d_y;

	int size = ARRAY_SIZE * sizeof(float);

	cudaMalloc((void **)&d_a, sizeof(float));
	cudaMalloc((void **)&d_x, size);
	cudaMalloc((void **)&d_y, size);

	x = (float*)malloc(size);
	random_floats(x, ARRAY_SIZE);
	y = (float*)malloc(size);
	yCompare = (float*)malloc(size);
	random_floats(y, ARRAY_SIZE);

	for(int i = 0; i<ARRAY_SIZE; ++i){
		yCompare[i] = y[i];
	}


	//printf("%x \n%x \n", yCompare, y);


	//cudaMemcpy(d_a, a, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);


	// HOST
	printf("Computing SAXPY on the CPU...");
	saxpy_host(a,x,yCompare);
	printf(" Done!\n");


	/*for(int i=0; i < ARRAY_SIZE; i++){
		//std::cout << (*yCompare+i) << " ";
	}
	printf("\n");
*/
	// DEVICE
	printf("Computing SAXPY on the GPU...");
	saxpy <<< BLOCKS, TPB >>>(a, d_x, d_y);
	cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
	printf(" Done!\n");


	for(int i=0; i < ARRAY_SIZE; i++){
		//std::cout << (*y+i) << " ";
	}	



//	*y = *y+1; // test for comparison implementation
	std::cout << "\n" << "error is: " << compareVec(y, yCompare);
	printf("\nVectors are equal: %s \n", compareVec(y, yCompare) < 0.01 ? "yes" : "no");


	/*if( yCompare == y){
		printf("de pekar på samma \n");
	}
	else {
		std::cout << "de pekar inte på samma";
	}
*/
	cudaError_t cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		printf("kernel launch failed with error \"%s\".\n",
			cudaGetErrorString(cudaerr));
	} else {
		printf("\ndone");
	}
	return 0;
}