//******************************************************************************
//  Author: Zachariah Bryant                                                   *
//  Function: Creates a 4-D or lattice that holds an integer at each site.     *
//            - For a 4-D lattice set first initialization to 3-D size and     *
//                the second parameter to the 4th dimension size.              *
//            - For a 3-D lattice just set the first initialization            *
//                parameter to the desired 3-D cube size.                      *
//                                                                             *
//******************************************************************************

#ifndef LATTICE_H
#define LATTICE_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif


class lattice{

private:
  int* latt;
  int xyzsize;
  int tsize;

  //Returns 1D location using the 4D parameters - PRIVATE
  CUDA_CALLABLE_MEMBER int GetLocation(int x, int y, int z, int t);

public:

  //Constructor
  CUDA_CALLABLE_MEMBER lattice(int xyzSizes, int tSize);

  //Destructor
  CUDA_CALLABLE_MEMBER ~lattice();

  //Makes new lattice for memory
  CUDA_CALLABLE_MEMBER void NewLattice(int xyzSizes, int tSize);

  //Returns the lattice value at a given location
  CUDA_CALLABLE_MEMBER int ReturnLocation(int x, int y, int z, int t);

  //Changes the value of lattice at given location.
  CUDA_CALLABLE_MEMBER void SetLocation(int x, int y, int z, int t, int newData);

  //Initializes all lattice sites to the passed data
  CUDA_CALLABLE_MEMBER void Initialize(int set);

  //Averages over the whole lattice
  //Note: This will only work for simple data types such as int, double, etc.
  CUDA_CALLABLE_MEMBER  double AverageLattice();



};

#endif
