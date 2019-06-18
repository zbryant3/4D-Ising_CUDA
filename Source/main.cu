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


//Header for gnuplot
#include "./Headers/gnuplot-iostream.h"


#include "./Headers/lattice.cuh"

using namespace std;


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 80 //Must be multiple of 8 for now



__global__ void test(){
  lattice test(4);
};


//**********************
//    Main Function    *
//**********************
int main()
{
  test<<<10,1>>>();

  return 0;

}
