//******************************************************************************
//  Author: Zachariah Bryant                                                   *
//  Function: Creates a Ising object to perform monte carlo simulations        *
//            with the NVIDIA GPU.                                             *
//                                                                             *
//******************************************************************************

#ifndef ISING_H
#define ISING_H

#include <iostream>
#include <vector>
#include <random>

#include "lattice.cuh"

using namespace std;

struct variables{
  //Lattice variables
  int size;
  double j;
  double beta;
  double h;

  //Constructor and destructor defined in .cu
  variables(int, double, double, double);
};


//**********************************
//    Class for CPU Ising model    *
//**********************************
class ising_model
{
private:
  variables *host_mem;
  variables *gpu_mem;
  lattice *host_major_lattice;
  lattice *gpu_major_lattice;


  //Equilibrates sublattices on the gpu
  //__global__ void GPU_Equilibriate(std::vector<std::vector<int>>);


public:

  //Constructor - Creates the lattice in the host memory and the device memory
  //Input Parameters - Lattice Dimension Size, J, beta, h
  __host__ ising_model(int, double, double, double);


  //Destructor - Deletes dynamically allocated memory
  __host__ ~ising_model();


  //Gets the Average Spin of the Lattice
  __host__ double AverageSpin();


  //Sets the value of beta in data
  __host__ void SetBeta(double newbeta);


  //Equilibrates the lattice by envoking a GPU kernal
  __host__ void Equilibrate();


};

#endif
