#include "./Headers/gpu_ising.cuh"
#include "./Headers/lattice.cuh"

//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h> //For testing


//*****************************
//      Private Functions     *
//*****************************


/**
 * Looks down in a given dimension, with periodic conditions
 * @param  loc - Current location on the given dimension
 * @return     - Returns an integer
 */
__device__ int gpu_Ising::LookDown(int loc){

        if((loc - 1) < 0)
                return (*LatticeSize - 1);
        else
                return (loc - 1);
};


/**
 * Looks up in a given dimension, with periodic conditions
 * @param  loc - Current location on the given dimension
 * @return     - Returns an integer
 */
__device__ int gpu_Ising::LookUp(int loc){

        if( (loc + 1) >= *LatticeSize)
                return 0;
        else
                return (loc + 1);
};



/**
 * Populates the sublattice based on the major lattice
 */
__device__ void gpu_Ising::PopulateSubLattice(){


        //Find the location on the shared memory lattice
        int minorX = threadIdx.x + 1;
        int minorY = threadIdx.y + 1;
        int minorZ = threadIdx.z + 1;
        int minorT = 1;

        //Find the thread location on the major lattice
        int majorX = threadIdx.x;
        int majorY = threadIdx.y + blockIdx.x * blockDim.y;
        int majorZ = threadIdx.z + blockIdx.y * blockDim.z;
        int majorT = blockIdx.z;


        //Fill the normal spots
        SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)]
                = Lattice[MajLocation(majorX, majorY, majorZ, majorT, *LatticeSize)];

        //Fill looking up in the T direction
        SubLattice[SubLocation(minorX, minorY, minorZ, minorT + 1, *LatticeSize)]
                = Lattice[MajLocation(majorX, majorY, majorZ, LookUp(majorT), *LatticeSize)];

        //Fill looking Down in the T direction
        SubLattice[SubLocation(minorX, minorY, minorZ, minorT - 1, *LatticeSize)]
                = Lattice[MajLocation(majorX, majorY, majorZ, LookDown(majorT), *LatticeSize)];

        __syncthreads();

        //Fill looking up in the X direction
        if(minorY == blockDim.y && minorZ == blockDim.z) {

                SubLattice[SubLocation(minorX + 1, minorY, minorZ, minorT, *LatticeSize)]
                        = Lattice[MajLocation(LookUp(majorX), majorY, majorZ, majorT, *LatticeSize)];

        }

        //Fill looking down in the X direction
        if(minorY == 1 && minorZ == 1) {

                SubLattice[SubLocation(minorX - 1, minorY, minorZ, minorT, *LatticeSize)]
                        = Lattice[MajLocation(LookDown(majorX), majorY, majorZ, majorT, *LatticeSize)];

        }

        __syncthreads();
        //Fill looking up in the Y direction
        if(minorX == blockDim.x && minorZ == blockDim.z) {

                SubLattice[SubLocation(minorX, minorY + 1, minorZ, minorT, *LatticeSize)]
                        = Lattice[MajLocation(majorX, LookUp(majorY), majorZ, majorT, *LatticeSize)];

        }

        //Fill looking down in the Y direction
        if(minorX == 1 && minorZ == 1) {

                SubLattice[SubLocation(minorX, minorY + 1, minorZ, minorT, *LatticeSize)]
                        = Lattice[MajLocation(majorX, LookDown(majorY), majorZ, majorT, *LatticeSize)];

        }

        __syncthreads();

        //Fill looking up in the Z direction
        if(minorX == blockDim.x && minorY == blockDim.y) {

                SubLattice[SubLocation(minorX, minorY, minorZ + 1, minorT, *LatticeSize)]
                        = Lattice[MajLocation(majorX, majorY, LookUp(majorZ), majorT, *LatticeSize)];

        }

        //Fill looking down in the Z direction
        if(minorX == blockDim.x && minorY == blockDim.y) {

                SubLattice[SubLocation(minorX, minorY, minorZ - 1, minorT, *LatticeSize)]
                        = Lattice[MajLocation(majorX, majorY, LookDown(majorZ), majorT, *LatticeSize)];

        }

        __syncthreads();

};


//Equilibrate the 3D segments of the Lattice
__device void ThreeDEquilibrate(){

  //Find the location on the shared memory lattice
  int minorX = threadIdx.x + 1;
  int minorY = threadIdx.y + 1;
  int minorZ = threadIdx.z + 1;
  int minorT = 1;

};




//****************************
//      Public Functions     *
//****************************





/**
 * Constructor for the GPU Ising Model object
 * @param  setConstants - Memory location of the lattice constants
 * @param  setLattice   - Memory location of the major lattice
 * @param  sharedlatt   - Memory location of the per-block(shared) lattice
 * @param  size       - Size of the sub lattice
 */
__device__ gpu_Ising::gpu_Ising(int *size, double *setbeta,
                                double *setj, double *seth,
                                int *SetLattice, int *SetSubLatt)
{
        LatticeSize = size;
        beta = setbeta;
        j = setj;
        h = seth;

        Lattice = SetLattice;
        SubLattice = SetSubLatt;
};


/**
 * Each thread equilibrates the lattice
 */
__device__ void gpu_Ising::Equilibrate(){

        PopulateSubLattice();

        //Checkerboard pattern for 4D (ie odd/even T locations equilibrate)
        int remainder = blockIdx.z%2;

        //Even T locations
        if(remainder == 0){
          ThreeDEquilibrate();
        }
        __syncthreads();

        if(remainder == 1){
          ThreeDEquilibrate();
        }


        //Odd T locations
        /*
           //Even checkerboard pattern for blocks
           if(remainder == 0){
           ThreadEquilibriate(gpu_mem, sub_lattice);
           }
           __syncthreads();


           //Odd Checkerboard pattern for blocks
           if(remainder == 1){
           ThreadEquilibriate(gpu_mem, sub_lattice);
           }
           __syncthreads();
         */
};
