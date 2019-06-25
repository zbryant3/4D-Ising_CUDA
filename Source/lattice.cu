#include "./Headers/lattice.cuh"


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <stdio.h> //For testing

/*
   //Makes new lattice for memory
   CUDA_CALLABLE_MEMBER void lattice::NewLattice(int xyzSizes, int tSize){
   delete[] latt;
   xyzsize = xyzSizes;
   tsize = tSize;
   latt = new int[xyzsize*xyzsize*xyzsize*tsize];
   };






   //Returns the lattice value at a given location
   CUDA_CALLABLE_MEMBER int lattice::ReturnLocation(int x, int y, int z, int t){
   return latt[GetLocation(x, y, z, t)];
   };



   //Changes the value of lattice at given location.
   CUDA_CALLABLE_MEMBER void lattice::SetLocation(int x, int y, int z, int t, int newData){
   latt[GetLocation(x,y,z,t)] = newData;
   };

 */



/**
 * Returns a 1D array location for the 4D lattice
 * @param  x       - X location
 * @param  y       - Y location
 * @param  z       - Z location
 * @param  t       - T location
 * @param  xyzsize - Size of each dimension
 * @return         - Integer for array location
 */
CUDA_CALLABLE_MEMBER int MajLocation(int x, int y, int z, int t, int LatticeSize){

        return (x + y*LatticeSize + z*LatticeSize*LatticeSize + t*LatticeSize*LatticeSize*LatticeSize);
};


CUDA_CALLABLE_MEMBER int SubLocation(int x, int y, int z, int t, int LatticeSize){

        return (x + y*(LatticeSize + 2) + z*(LatticeSize + 2)*(LatticeSize/2 + 2)
        + t*(LatticeSize + 2)*(LatticeSize/2 + 2)*(LatticeSize/2 + 2));
};




/**
 * Initializes the 4D lattice sites to the passed integer
 * @param lattice - Array that the lattice resides on
 * @param size    - Size of the lattice
 * @param setData - Integer to set all lattice sites to
 */
CUDA_CALLABLE_MEMBER void Initialize(int *lattice, int size, int setData){

        for(int i = 0; i < size; i++)
                for(int j = 0; j < size; j++)
                        for(int k = 0; k < size; k++)
                                for (int z = 0; z < size; z++)
                                        lattice[MajLocation(i,j,k,z,size)] = 1;

};


/*
   //Averages over the whole lattice
   //Note: This will only work for simple data types such as int, double, etc.
   CUDA_CALLABLE_MEMBER  double lattice::AverageLattice(){
   double total{0};

   for(int i = 0; i < xyzsize; i++)
    for(int j = 0; j < xyzsize; j++)
      for(int k = 0; k < xyzsize; k++)
        for (int z = 0; z < tsize; z++)
          total += ReturnLocation(i, j, k, z);

   return total/(xyzsize*xyzsize*xyzsize*tsize);

   }; */
