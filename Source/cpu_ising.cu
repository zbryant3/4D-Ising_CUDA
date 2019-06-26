//User made headers
#include "./Headers/cpu_ising.cuh"

#include "./Headers/gpu_ising.cuh"
#include "./Headers/lattice.cuh"


#include <cuda.h>
#include <curand_kernel.h>

#include <iostream>
#include <stdio.h>
using namespace std;




/**
 * GPU kernal for initiating equilibration of the lattice
 * @param gpu_const   - Memory location for constants of lattice
 * @param gpu_lattice - Memory location for the major lattice
 * @param memsize     - Size of the sub lattice
 */
__global__ void GPU_Equilibriate(int *LatticeSize, double *beta, double *j, double *h, int *Lattice){

        //Shared sublattice memory for each block to share among the threads
        extern __shared__ int sub_lattice[];

        gpu_Ising ising(LatticeSize, beta, j, h, Lattice, sub_lattice);
        ising.Equilibrate();
};



/**
 * Constructor - Creates the lattice in the host memory and the device memory
 * @param  lattSize - Lattice size for the ising model
 * @param  j        - j parameter for the model
 * @param  beta     - beta parameter for temperature (1/T)
 * @param  h        - magnetic field
 */
__host__ ising_model::ising_model(int lattSize, double jay, double betaset, double hset){

        //Construct constants in host
        LatticeSize = lattSize;
        j = jay;
        beta = betaset;
        h = hset;
        ArrSize = LatticeSize*LatticeSize*LatticeSize*LatticeSize;


        //Allocate gpu memory and then copy from host memory
        cudaMalloc((void**)&GPU_LatticeSize, sizeof(int));
        cudaMalloc((void**)&GPU_j, sizeof(double));
        cudaMalloc((void**)&GPU_beta, sizeof(double));
        cudaMalloc((void**)&GPU_h, sizeof(double));

        cudaMemcpy(GPU_LatticeSize, &LatticeSize, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_j, &j, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_h, &h, sizeof(double), cudaMemcpyHostToDevice);

        //Construct major lattice for the host and gpu memory
        host_lattice = new int[ArrSize];
        Initialize(host_lattice, LatticeSize, 1);

        cudaMalloc((void**)&gpu_lattice, ArrSize*sizeof(int));
        cudaMemcpy(gpu_lattice, host_lattice, ArrSize*sizeof(int), cudaMemcpyHostToDevice);

};



/**
 * Destructor - Deletes dynamically allocated memory
 */
__host__ ising_model::~ising_model(){
        cudaFree(GPU_h);
        cudaFree(GPU_j);
        cudaFree(GPU_beta);
        cudaFree(GPU_LatticeSize);

        delete host_lattice;
        cudaFree(gpu_lattice);
};




/**
 * Averages the spin of the lattice
 * @return - average spin
 */
__host__ double ising_model::AverageSpin(){

        double average{0};
        for(int i = 0; i < ArrSize; i++)
                average += host_lattice[i];

        return (average/ArrSize);
};




/**
 * Sets the value of beta in the memory of the host and the gpu
 * @param newbeta - the new value of beta
 */
__host__ void ising_model::SetBeta(double newbeta){
        beta = newbeta;
        cudaMemcpy(GPU_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);
};




/**
 * Initiates equilibration of the lattice on the GPU
 */
__host__ void ising_model::Equilibrate(){

        //Specify dimensions of the sub_lattice
        dim3 threads(LatticeSize, LatticeSize/4, LatticeSize/4);
        dim3 blocks(4, 4, LatticeSize);


        /* memsize is the size of the shared memory lattice for each block
           Dimensions:     X                Y                    Z                T                    */
        int memsize = (LatticeSize + 2)*(LatticeSize/4 + 2)*(LatticeSize/4 + 2) * 3;


        //Copy host lattice to gpu
        cudaMemcpy(gpu_lattice, host_lattice, ArrSize*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(GPU_beta, &beta, sizeof(double), cudaMemcpyHostToDevice);


        //Equilibrate lattice on GPU
        GPU_Equilibriate<<<blocks, threads, memsize*sizeof(int)>>>(
                GPU_LatticeSize, GPU_beta, GPU_j, GPU_h, gpu_lattice);
        cudaDeviceSynchronize();


        //Copy Equilibrated lattice back to host
        cudaMemcpy(host_lattice, gpu_lattice, ArrSize*sizeof(int), cudaMemcpyDeviceToHost);

};
