//******************************************************************************
//  Author: Zachariah Bryant                                                   *
//  Function: Creates a Ising object to perform monte carlo simulations        *
//            with the NVIDIA GPU.                                             *
//                                                                             *
//******************************************************************************


#ifndef GPU_ISING_H
#define GPU_ISING_H

#include <iostream>

#include "lattice.cuh"
#include "cpu_ising.cuh"

using namespace std;



class gpu_Ising
{
private:
  int *LatticeSize;
  double *beta;
  double *j;
  double *h;

  int *Lattice;
  int *SubLattice;


  //Looks down in a given dimension, if the dimension is too small it returns
  // the size of the lattice -1, giving the lattice a periodic nature
  __device__ int LookDown(int);


  //Looks up in a given dimnension, if the dimension is too large it returns a
  //0 - giving our lattice a periodic nature
  __device__ int LookUp(int);


  //Populates the sublattice from the major lattice
  __device__ void PopulateSubLattice();


  //Equilibrate the 3D segments of the Lattice
  __device void ThreeDEquilibrate();

  //Gets the difference in energy if spin is changed
  //__device__ double EnergyDiff(variables, int*);


  //Returns the Boltzmann distribution of a given energy difference
  //_device__ float BoltzmannDist(variables, double);


  //Equilibriates a given sublattice
  //__device__ void ThreadEquilibriate(variables, int*);


public:

  //Constructor
  __device__ gpu_Ising(int*, double*, double*, double*, int*, int*);

  //Each thread will equilibrate the lattice
  __device__ void Equilibrate();



};









/*

//**********************************
//    Class for GPU Ising model    *
//**********************************
class gpu_ising_model
{
private:
  variables *host_mem;
  variables *gpu_mem;
  int *host_major_lattice;
  int *gpu_major_lattice;


  //Equilibrates sublattices on the gpu
  //__global__ void GPU_Equilibriate(std::vector<std::vector<int>>);







public:

  //Constructor - Creates the lattice in the host memory and the device memory
  //Input Parameters - Lattice Dimension,
  __host__ ising_model(int, double, double, double);


  //Destructor - Deletes dynamically allocated memory
  __host__ ~ising_model();


  //Gets the average spin of the lattice using the gpu
  __host__ double AverageSpin();

  __host__ void SetBeta(double newbeta){
    host_mem->beta = newbeta;
    cudaMemcpy(gpu_mem, host_mem, sizeof(variables), cudaMemcpyHostToDevice);
  }


  //Equilibrates the lattice
  __host__ void Equilibrate();
};

*/

#endif
