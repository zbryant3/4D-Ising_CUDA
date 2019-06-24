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
__global__ void GPU_Equilibriate(variables *gpu_const, lattice *gpu_lattice, int memsize){

  //Shared sublattice memory for each block to share among the threads
  extern __shared__ int sub_lattice[];

  gpu_Ising ising(gpu_const, gpu_lattice, sub_lattice, memsize);
  ising.Equilibrate();
};


/**
 * Constructor to hold various variables
 * @param  lattSize - Lattice size for the ising model
 * @param  j        - j parameter for the model
 * @param  beta     - beta parameter for temperature (1/T)
 * @param  h        - magnetic field
 */
variables::variables(int lattSize, double setJ, double setBeta, double setH){
  size = lattSize;
  j = setJ;
  beta = setBeta;
  h = setH;
};


 /**
  * Constructor - Creates the lattice in the host memory and the device memory
  * @param  lattSize - Lattice size for the ising model
  * @param  j        - j parameter for the model
  * @param  beta     - beta parameter for temperature (1/T)
  * @param  h        - magnetic field
  */
__host__ ising_model::ising_model(int lattSize, double j, double beta, double h){

  //Construct constants in host and gpu memory
  host_const = new variables(lattSize, j, beta, h);
  cudaMalloc(&gpu_const, sizeof(variables));
  cudaMemcpy(gpu_const, host_const, sizeof(variables), cudaMemcpyHostToDevice);

  //Construct major lattice for the host and gpu memory
  host_lattice = new lattice(lattSize, lattSize);
  host_lattice->Initialize(1);

  cudaMalloc(&gpu_lattice, sizeof(lattice));
  cudaMemcpy(gpu_lattice, host_lattice, sizeof(lattice), cudaMemcpyHostToDevice);

};




/**
 * Destructor - Deletes dynamically allocated memory
 */
__host__ ising_model::~ising_model(){
  delete host_const;
  cudaFree(gpu_const);

  delete host_lattice;
  cudaFree(gpu_lattice);
};




/**
 * Averages the spin of the lattice
 * @return - average spin
 */
__host__ double ising_model::AverageSpin(){
  return host_lattice->AverageLattice();
};



/**
 * Sets the value of beta in the memory of the host and the gpu
 * @param newbeta - the new value of beta
 */
__host__ void ising_model::SetBeta(double newbeta){
  host_const->beta = newbeta;
  cudaMemcpy(gpu_const, host_const, sizeof(variables), cudaMemcpyHostToDevice);
};




/**
 * Initiates equilibration of the lattice on the GPU
 */
__host__ void ising_model::Equilibrate(){

  //Set size for gpu calculations
  int size = host_const->size;

  //Specify dimensions of the sub_lattice
  dim3 threads(size, size/2, size/2);
  dim3 blocks(2, 2, size);


  /* memsize is the size of the shared memory lattice for each block
     Dimensions:     X          Y             Z        T                    */
  int memsize = (size + 2)*(size/2 + 2)*(size/2 + 2) * 3;


  //Copy host lattice to gpu
  cudaMemcpy(gpu_lattice, host_lattice, sizeof(host_lattice), cudaMemcpyHostToDevice);


  //Equilibrate lattice on GPU
  GPU_Equilibriate<<<blocks, threads, memsize*sizeof(int)>>>(gpu_const, gpu_lattice, memsize);
  cudaDeviceSynchronize();


  //Copy Equilibrated lattice back to host
  cudaMemcpy(host_lattice, gpu_lattice, sizeof(gpu_lattice), cudaMemcpyDeviceToHost);

};
