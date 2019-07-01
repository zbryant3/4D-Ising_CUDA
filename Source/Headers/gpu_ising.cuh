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

//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>

using namespace std;



class gpu_Ising
{
private:
  int *LatticeSize;
  double *beta;
  double *j;
  double *h;

  int minorX;
  int minorY;
  int minorZ;
  int minorT;

  int majorX;
  int majorY;
  int majorZ;
  int majorT;

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
  __device__ void ThreeDEquilibrate();

  //Gets the difference in energy if spin is changed
  __device__ double EnergyDiff(int, int);


  //Returns the Boltzmann distribution of a given energy difference
  __device__ double BoltzmannDist(double);

  //Returns the sum of the neigherbors spin at a lattice site
  __device__ double SumNeighborSpin();

public:

  //Constructor
  __device__ gpu_Ising(int*, double*, double*, double*, int*, int*);

  //Each thread will equilibrate the lattice
  __device__ void Equilibrate();



};


#endif
