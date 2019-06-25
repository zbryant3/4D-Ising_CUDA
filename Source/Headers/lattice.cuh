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




//Returns 1D location using the 4D parameters - PRIVATE
CUDA_CALLABLE_MEMBER int MajLocation(int x, int y, int z, int t, int xyzsize);

//Returns 1D location using the 4D parameters - PRIVATE
CUDA_CALLABLE_MEMBER int SubLocation(int x, int y, int z, int t, int xyzsize);


//Initializes all lattice sites to the passed data
CUDA_CALLABLE_MEMBER void Initialize(int *, int, int);



#endif
