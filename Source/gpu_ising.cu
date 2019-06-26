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


/**
 * Gets the difference in energy based on neighboring spins
 * @param  old_spin - The old spin at the lattice site
 * @param  new_spin - The new spin at the lattice site
 * @return          - The energy difference
 */
__device__ double gpu_Ising::EnergyDiff(int old_spin, int new_spin){

        double sumNeighborSpin{0};

        //Look at neighbors in the X direction
        sumNeighborSpin += SubLattice[SubLocation(minorX + 1, minorY, minorZ, minorT, *LatticeSize)];
        sumNeighborSpin += SubLattice[SubLocation(minorX - 1, minorY, minorZ, minorT, *LatticeSize)];

        //Look at neighbors in the Y direction
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY + 1, minorZ, minorT, *LatticeSize)];
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY - 1, minorZ, minorT, *LatticeSize)];

        //Look at neighbors in the Z direction
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY, minorZ + 1, minorT, *LatticeSize)];
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY, minorZ - 1, minorT, *LatticeSize)];

        //Look at neighbors in the T direction
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY, minorZ, minorT + 1, *LatticeSize)];
        sumNeighborSpin += SubLattice[SubLocation(minorX, minorY, minorZ, minorT - 1, *LatticeSize)];

        return ((-1)*(*j)*sumNeighborSpin*(old_spin - new_spin)
                + (-1)*(*h)*(old_spin - new_spin));

};




//Returns the Boltzmann distribution of a given energy difference
__device__ double gpu_Ising::BoltzmannDist(double energydiff){
        return expf((-1)*(*beta)*energydiff);
};


//Equilibrate the 3D segments of the Lattice
__device__ void gpu_Ising::ThreeDEquilibrate(){

        int old_spin = SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)];
        int new_spin = (-1)*old_spin;
        double energydiff{0};

        int remainder = (minorX + minorY + minorZ)%2;

        int tid = MajLocation(threadIdx.x, (threadIdx.y + blockIdx.x * blockDim.y),
                              (threadIdx.z + blockIdx.y * blockDim.z), blockIdx.z, *LatticeSize);


        curandState_t rng;
        curand_init(clock64(), tid, 0, &rng);

        //Even 3D threads
        if(remainder == 0) {
                energydiff = EnergyDiff(old_spin, new_spin);

                //If the energy difference is lower or based on a
                //random probability accept the new spin.
                if(energydiff <= 0) {
                        SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)] = new_spin;
                } else if(curand_uniform(&rng) < BoltzmannDist(energydiff)) {
                        SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)] = new_spin;
                }
        }
        __syncthreads();



        //Odd 3D threads
        if(remainder == 1) {
                energydiff = EnergyDiff(old_spin, new_spin);

                //If the energy difference is lower or based on a
                //random probability accept the new spin.
                if(energydiff <= 0) {
                        SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)] = new_spin;
                } else if(curand_uniform(&rng) < BoltzmannDist(energydiff)) {
                        SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)] = new_spin;
                }
        }
        __syncthreads();

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

        //Find the location on the shared memory lattice
        minorX = threadIdx.x + 1;
        minorY = threadIdx.y + 1;
        minorZ = threadIdx.z + 1;
        minorT = 1;


        //Find the thread location on the major lattice
        majorX = threadIdx.x;
        majorY = threadIdx.y + blockIdx.x * blockDim.y;
        majorZ = threadIdx.z + blockIdx.y * blockDim.z;
        majorT = blockIdx.z;
};


/**
 * Each thread equilibrates the lattice
 */
__device__ void gpu_Ising::Equilibrate(){

        PopulateSubLattice();

        //Checkerboard pattern for 4D (ie odd/even T locations equilibrate)
        int remainder = blockIdx.z%2;

        //Even T dimension locations
        if(remainder == 0) {
                ThreeDEquilibrate();
        }
        __syncthreads();

        //Odd T dimension locations
        if(remainder == 1) {
                ThreeDEquilibrate();
        }
        __syncthreads();

        //Fill the normal spots
        Lattice[MajLocation(majorX, majorY, majorZ, majorT, *LatticeSize)]
                = SubLattice[SubLocation(minorX, minorY, minorZ, minorT, *LatticeSize)];

};
