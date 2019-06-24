#include "./Headers/lattice.cuh"


#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <stdio.h> //For testing


//Constructor
CUDA_CALLABLE_MEMBER lattice::lattice(int set3dSize, int settSize){
  xyzsize = set3dSize;
  tsize = settSize;
  latt = new int[xyzsize*xyzsize*xyzsize*tsize];
};



//Destructor
CUDA_CALLABLE_MEMBER lattice::~lattice(){
  delete[] latt;
};

//Makes new lattice for memory
CUDA_CALLABLE_MEMBER void lattice::NewLattice(int xyzSizes, int tSize){
  delete[] latt;
  xyzsize = xyzSizes;
  tsize = tSize;
  latt = new int[xyzsize*xyzsize*xyzsize*tsize];
};



//Returns 1D location using the 4D parameters - PRIVATE
CUDA_CALLABLE_MEMBER int lattice::GetLocation(int x, int y, int z, int t){
  int index = x + y*xyzsize + z*xyzsize*xyzsize + t*xyzsize*xyzsize*xyzsize;
  return index;
};



//Returns the lattice value at a given location
CUDA_CALLABLE_MEMBER int lattice::ReturnLocation(int x, int y, int z, int t){
  return latt[GetLocation(x, y, z, t)];
};



//Changes the value of lattice at given location.
CUDA_CALLABLE_MEMBER void lattice::SetLocation(int x, int y, int z, int t, int newData){
  latt[GetLocation(x,y,z,t)] = newData;
};



//Initializes all lattice sites to the passed data
CUDA_CALLABLE_MEMBER void lattice::Initialize(int setData){

  for(int i = 0; i < xyzsize; i++){
    for(int j = 0; j < xyzsize; j++)
      for(int k = 0; k < xyzsize; k++)
        for (int z = 0; z < tsize; z++)
          SetLocation(i, j, k, z, setData);
        }
};



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

};
