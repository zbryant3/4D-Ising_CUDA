#include "./Headers/gpu_ising.cuh"
#include "./Headers/lattice.cuh"

//CUDA enabled random number generator
#include <cuda.h>
#include <curand_kernel.h>

#include <stdio.h> //For testing


//*****************************
//      Private Functions     *
//*****************************





//Looks down in a given dimension, if the dimension is too small it returns
// the size of the lattice -1, giving the lattice a periodic nature
__device__ int gpu_Ising::LookDown(int loc){

        if((loc - 1) < 0)
                return (constants->size - 1);
        else
                return (loc - 1);
};


//Looks up in a given dimnension, if the dimension is too large it returns a
//0 - giving our lattice a periodic nature
__device__ int gpu_Ising::LookUp(int loc){

        if( (loc + 1) >= constants->size)
                return 0;
        else
                return (loc + 1);
};



//Returns 1D array location using the 4D parameters
__device__ int gpu_Ising::GetLoc(int x, int y, int z, int t){
        return x + y * (constants->size / 2) + z * (constants->size / 2) * (constants->size/2) + t * (constants->size / 2) * (constants->size / 2) * 3;
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

        int test = lattice->ReturnLocation(majorX, majorY, majorZ, majorT);
        //sharedlattice[GetLoc(minorX, minorY, minorZ, minorT)]
        //        = lattice->ReturnLocation(majorX, majorY, majorZ, majorT);


        //Fill looking up in the T direction
        //sharedlattice[GetLoc(minorX, minorY, minorZ, minorT + 1)]
        //        = lattice->ReturnLocation(majorX, majorY, majorZ, LookUp(majorT));

        //Fill looking Down in the T direction
        //sharedlattice[GetLoc(minorX, minorY, minorZ, minorT - 1)]
        //        = lattice->ReturnLocation(majorX, majorY, majorZ, LookDown(majorT));






        //Populate sub lattice to major lattice values
        //sub_lattice.SetLocation(majorX, majorY, majorZ, 1, lattice->GetLocation())

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
__device__ gpu_Ising::gpu_Ising(variables *setConstants,
                                lattice::lattice *setLattice,
                                int *sharedlatt,
                                int size)
{
        constants = setConstants;
        lattice = setLattice;
        sharedlattice = sharedlatt;
        sub_Size = size;
};


/**
 * Each thread equilibrates the lattice
 */
__device__ void gpu_Ising::Equilibrate(){

        PopulateSubLattice();


        //int remainder = (blockIdx.y  + blockIdx.x) % 2;

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
