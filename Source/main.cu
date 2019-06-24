//*****************************************************
// Usage: Performs Ising model simulations utilizing  *
//  monte carlo calculations performed on the GPU.    *
//                                                    *
// Author: Zachariah Bryant                           *
//*****************************************************


//**************
//   Headers   *
//**************
#include <iostream>
#include <fstream>
#include <string.h>

//Enables use of gnuplot from file
#include "./Headers/gnuplot-iostream.h"

//Contains class wrap for ising model to be performed on the gpu
#include "./Headers/cpu_ising.cuh"


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 16 //Must be multiple of 8 for now
#define J .5
#define BETA 0.2
#define H 0.1


//**********************
//    Main Function    *
//**********************
int main()
{
  ising_model test(LATTSIZE, J, BETA, H);
  test.Equilibrate();

  return 0;

}
