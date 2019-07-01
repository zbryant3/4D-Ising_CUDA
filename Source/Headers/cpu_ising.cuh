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


//Kernal for GPU Equilibration
__global__ void GPU_Equilibriate(int*, double*, double*, double*, int*);


//**********************************
//    Class for CPU Ising model    *
//**********************************
class ising_model
{
private:
//Host constant variables
int LatticeSize;
double j;
double beta;
double h;
int ArrSize;

//GPU constant variables
int *GPU_LatticeSize;
double *GPU_j;
double *GPU_beta;
double *GPU_h;

//Host lattice
int *host_lattice;
int *gpu_lattice;

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

//Gets the correlation of the lattice sites
__host__ double Correlation();


};

#endif
