#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>

#include <random>
#include <iostream>
#include <math.h>

#include <iomanip>
#include <string>
#include <map>
#include <cstdlib>
#include <ctime>

#include <fstream>

#include <sys/time.h>

#include <stdio.h>

//#define NUM_PARTICLES 1e7
#define NUM_ITERATIONS 5000
#define dt 1.0f


__host__ __device__ float3 operator+(const float3 &a, const float3 &b) {
	return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__host__ __device__ float3 operator*(const float &a, const float3 &b) {
	return make_float3(a*b.x, a*b.y, a*b.z);
}

__host__ __device__ float3 operator*(const float3 &a, const float &b) {
	return make_float3(b*a.x, b*a.y, b*a.z);
}


double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}



struct Particle {
	float3 position;
	float3 velocity;
};

__global__ void timeStep(Particle *par, int TPB){

	int index  = blockIdx.x * TPB + threadIdx.x;

	par[index].velocity = par[index].velocity + (-1.0f * par[index].position * dt); 
	
	par[index].position = par[index].position + par[index].velocity * dt; 

}


void timeStep_cpu(Particle *par, int NUM_PARTICLES){


	for(int i = 0; i<NUM_PARTICLES; ++i){
		par[i].velocity = par[i].velocity + (-1.0f * (par[i].position) * dt); 
	}


	for(int i = 0; i<NUM_PARTICLES; ++i){
		par[i].position = par[i].position + (par[i].velocity) * dt; 
	}

}



int main(int argc, char **argv) {


	int TPB = atoi(argv[1]);
	int NUM_PARTICLES = atoi(argv[2]);


	int block_sizes[] = {16, 32, 64, 128, 256};
	int num_particles[] = {10, 100, 1000, 10000, 100000};

	//int block_array_len = 5;
	//int num_particles_len = 5;


	int block_array_len = 1;
	int num_particles_len = 1;



	Particle *particles;
	Particle *d_particles;
	Particle *particlesCompare;
	

	std::ofstream myfile("data.txt");


	for(int i = 0; i < block_array_len; ++i){
		for(int j = 0; j < num_particles_len; ++j){

			//std::cout << "TPB: " << TPB << "\n";
			//std::cout << "numP: " << NUM_PARTICLES << "\n";

			//TPB = block_sizes[i];
			//NUM_PARTICLES = num_particles[j];

			std::cout << "TPB: " << TPB << ", num particles: " << NUM_PARTICLES <<  "\n";
			
			int BLOCKS = (NUM_PARTICLES + TPB - 1)/TPB;


			//Particle  *particles = (Particle *)calloc(NUM_PARTICLES, sizeof(Particle));

			particles = (Particle*)malloc(sizeof(Particle)*NUM_PARTICLES);
			particlesCompare = (Particle*)malloc(sizeof(Particle)*NUM_PARTICLES);

			for(int k = 0; k < NUM_PARTICLES; ++k){
				particles[k].position = make_float3((float)rand()/(float)(RAND_MAX)*5.0f, (float)rand()/(float)(RAND_MAX)*5.0f, (float)rand()/(float)(RAND_MAX)*5.0f);
				particles[k].velocity = make_float3((float)rand()/(float)(RAND_MAX)*5.0f, (float)rand()/(float)(RAND_MAX)*5.0f, (float)rand()/(float)(RAND_MAX)*5.0f);
			}

			// Kopiera partiklarna in i compare
			for (int k = 0; k < NUM_PARTICLES; ++k)
			{
				particlesCompare[k].position = particles[k].position;
				particlesCompare[k].velocity = particles[k].velocity;
			} 
		/*	// verifiera att kopieringen ovan fungerar
			printf("ParticlesCompare:\nPosition: x=%f, y=%f, z=%f\n", particlesCompare[0].position.x, particlesCompare[0].position.y, particlesCompare[0].position.z);
			printf("Velocity: x=%f, y=%f, z=%f\n\n", particlesCompare[0].velocity.x, particlesCompare[0].velocity.y, particlesCompare[0].velocity.z);

			for (int i = 0; i < NUM_PARTICLES; ++i){
				printf("Position: x=%f, y=%f, z=%f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
				printf("Velocity: x=%f, y=%f, z=%f\n\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
			}

		*/
			int size = sizeof(Particle)*NUM_PARTICLES;
			cudaMalloc((void **)&d_particles, size);
				
			cudaMemcpy(d_particles, particles, size, cudaMemcpyHostToDevice);

			printf("Calculating on GPU... \n");

			double iStart = cpuSecond();
			for(int k = 0; k < NUM_ITERATIONS; ++k){
				timeStep <<< BLOCKS, TPB >>>(d_particles, TPB);
				cudaDeviceSynchronize();
			}
			double time_elapsed = cpuSecond() - iStart;

			printf("Done, elapsed time: %f s\n", time_elapsed );

				if (myfile.is_open()){
					myfile << "GPU " << TPB << " " << NUM_PARTICLES << " " << time_elapsed << "\n";
				}

			cudaMemcpy(particlesCompare, d_particles, size, cudaMemcpyDeviceToHost);
		/*
			printf("Particle values after timeStep:\n");
			for (int i = 0; i < NUM_PARTICLES; ++i){
				printf("Position: x=%f, y=%f, z=%f\n", particlesCompare[i].position.x, particlesCompare[i].position.y, particlesCompare[i].position.z);
				printf("Velocity: x=%f, y=%f, z=%f\n\n", particlesCompare[i].velocity.x, particlesCompare[i].velocity.y, particlesCompare[i].velocity.z);
			}
		*/

			printf("Calculating on CPU... \n");

			iStart = cpuSecond();
			for(int k = 0; k < NUM_ITERATIONS; ++k){
				timeStep_cpu(particles, NUM_PARTICLES);
			}
			time_elapsed = cpuSecond() - iStart;

			if (myfile.is_open()){
				myfile << "CPU 0 " << NUM_PARTICLES << " " << time_elapsed << "\n";
			}

			printf("Done, elapsed time: %f s\n", time_elapsed );

		/*	for (int i = 0; i < NUM_PARTICLES; ++i){
				printf("Position: x=%f, y=%f, z=%f\n", particles[i].position.x, particles[i].position.y, particles[i].position.z);
				printf("Velocity: x=%f, y=%f, z=%f\n\n", particles[i].velocity.x, particles[i].velocity.y, particles[i].velocity.z);
			}
		*/

		}

	}

	myfile.close();

	return 0;
}
