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
#include <vector>

//Enables use of gnuplot from file
#include "./Headers/gnuplot-iostream.h"

//Contains class wrap for ising model to be performed on the gpu
#include "./Headers/cpu_ising.cuh"


//**************************************
//   Definition of all the variables   *
//**************************************
#define LATTSIZE 16 //Must be multiple of 8 for now
#define J .5
#define H 0.5

#define EQSWEEPS 10000
#define CONFIGS 500
#define STARTTEMP 5
#define TEMPCHANGE -0.05
#define TEMPLIMIT 0.2


using namespace std;


double Average(vector<double> avgspin)
{
  double total{0};

  for(unsigned int i = 0; i < avgspin.size(); i++)
  total += avgspin[i];

  return total/avgspin.size();
}

double Standard_Deviation(vector<double> avgspin)
{
  double x{0};
  double y{Average(avgspin)};

  for(int i = 0; i < CONFIGS; i++)
  x += pow((avgspin[i] - y), 2);

  x = sqrt(x)/sqrt(CONFIGS*(CONFIGS-1));
  return x;
}


//**********************
//    Main Function    *
//**********************
int main()
{
  double temp{STARTTEMP};

  //Create a Ising Model object to perform operations on
  ising_model ising(LATTSIZE, J, (1/temp), H);

  ising.SetBeta(1/temp);

  /*            FOR CORRELATION
  fstream File;
  File.open("Corr_vs_Eq.dat", ios::out | ios::trunc);
  cout << "************************   EQUILIBRATING   ***********************\n";
  for(int i = 0; i < CONFIGS; i++){
    File << i << " " << ising.Correlation() << "\n";
    File.flush();

  }
  File.close();

  */





  //Files for logging data
  fstream File0, File1;

  //Thermalize lattice and log data
  File0.open("Spin_vs_Eq.dat", ios::out | ios::trunc );
  cout << "************************ THERMALIZING ************************ \n";
  for(int i = 0; i < EQSWEEPS; i++){
    File0 << i << " " << ising.AverageSpin() << "\n";
    File0.flush();

    ising.Equilibrate();
  }
  File0.close();



  //Iterate Temperature of the object and log the data
  vector<double> AvgSpin(CONFIGS);

  string name = "Spin_vs_T(J=";
  string Jay = to_string(J);
  Jay.resize(4); //Truncates J to 4 characters
  name += Jay;
  name += ").dat";
  File1.open( name, ios::out | ios::trunc );
  double average{0};

  //Calculation of avg. spin vs temperature, i.e. we change beta,
  cout << "****************** ITERATING TEMP ************************ \n";
  do
  {
    for(int i = 0; i < 100; i++){
      ising.Equilibrate();
    }

    //Collects the average spin based on how many CONFIGS desired
    for(int i = 0; i < CONFIGS; i++)
    {
      //Seperates configurations for measurement
      for(int j = 0; j < 2; j++){
        ising.Equilibrate();
      }

      //Measure avg. over all spins of a given configurations
      AvgSpin[i] = ising.AverageSpin();
    }

    average = Average(AvgSpin);
    File1 << temp << "  " << average << " " << Standard_Deviation(AvgSpin) <<  "\n";
    File1.flush();


    temp += TEMPCHANGE;
    ising.SetBeta(1/(temp));
    cudaDeviceSynchronize();
  }
  while(temp > TEMPLIMIT);

  File1.close();



  /*
  //Plots the data automatically
  Gnuplot gp;
  gp << "set title \"Average Spin vs T\" \n";
  gp << "set xlabel \" Temperature (1/Beta)\"\n";
  gp << "set ylabel \"Average Spin\" \n";
  gp << "plot [0:5] '"<< name <<"' using 1:2:3 w yerr\n";
  gp << "set term png \n";
  gp << "set output \"" << name << "\".png \n";
  gp << "replot\n";
  gp << "set term x11 \n";


  cout << "****** FINISHED ******** \n \n" ;
  */

  return 0;

}
