//
//  FILE:   main.cpp
//  MODULE: mdgpu ("Molecular Dynamics for the GPU")
//
//  DESCRIPTION:
//  File contains the main executive for running hard disk collision simulations.  Input is read from a file, time integration and collision updates of both phase space and tangent space vectors is performed, output is recorded to user-defined files and allocated memory at run-time is freed.
//
//  REFERENCE:
//  Dinius, J.          "Dynamical Properties of a Generalized Collision Rule for Multi-Particle Systems" Ph.D dissertation.
//  Brandes, T. et. al  "CPU vs. GPU - Performance comparison for the Gram-Schmidt Algorithm"
//
//  REVISION HISTORY:
//  Dinius, J.       Initial release                              09/07/14
//

// INCLUDES/DECLARATIONS
#include <iostream>
#include <string>
#include "utilities.h"
#include <cmath>
#include "stdlib.h"

// for inherited QR routine (see second reference)
#include "QR.hpp"
#include "cublas.h"

using namespace std;
// END INCLUDES/DECLARATIONS

// VARIABLES IN GLOBAL SCOPE
// END VARIABLES IN GLOBAL SCOPE

//
//  FUNCTION: main
//  MODULE:   mdgpu
//
//  DESCRIPTION:
//  Function is the main executive routine for generalized collision simulations.
//
//  INPUTS:
//  argc - standard form for C/C++ main routine for integer input (currently unused)
//  argv - standard form for C/C++ main routine for character array input (used for input filename to read initialization data from)
//
//  OUTPUTS:
//  0 - normal execution
//  1 - clean exit due to user-defined exception declared
//

//
//  REVISION HISTORY:
//  Dinius, J.       Created                              10/08/11
//
int main (int argc, const char * argv[])
{
    // DECLARATIONS FOR INPUT
    ifstream inData;          // file handle to input file
    string   dummyStr;        // dummy string for holding temporary data read-in from input file
    string   str1 ("*/");     // string for end of line comparison during input read
    int      nSteps;          // number of time steps to run simulation
    int      nModes;          // number of tangent vectors to write out to file
    int      nOutput;         // number of steps between successive writes to main output file
    int      i,j;             // loop counters (initialized once for convenience)
	int      processCLVs;     // REVISIT LATER
	int      nDisks;
	int      nOrtho;
	int      countCLV;
	float    density;
	float    stepSize;
	ofstream clvData;
	float    clvSettleTime;

    // END DECLARATIONS FOR INPUT

	//OPEN INPUT FILE
	inData.open(argv[1]);
    //inData.open("inputTest.dat"); //DEBUG
    
	if (!inData){
		// INPUT FILE COULD NOT BE OPENED, SO EXIT
		cerr << "Error: input file could not be opened.  Exiting..." << endl;
		exit(1);
	} // END if (!inData)
	
	// set precision for output to screen
	cout.precision(5);

	// DECLARATIONS FOR OUTPUT
	ofstream outData;                // file handle to output file
	char *outFile = new char [256];  // character array to hold output filename
	// END DECLARATIONS FOR OUTPUT

	// READ IN DATA FROM INPUT FILE AND OUTPUT VALUES TO SCREEN
	cout << "mdgpu: Read-in parameters:" << endl;
	inData >> stepSize;
	inData >> dummyStr;
	cout << "Stepsize = " << stepSize << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> nSteps;
	inData >> dummyStr;
	cout << "No. of Steps = " << nSteps << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> nOrtho;
	inData >> dummyStr;
	cout << "No. of Steps between Orthonormalization Operations = " << nOrtho << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> nOutput;
	inData >> dummyStr;
	cout << "No. of Steps between File Output = " << nOutput << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> nDisks;
	inData >> dummyStr;
	cout << "No. of Disks = " << nDisks << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> density;
	inData >> dummyStr;
	cout << "Density of disks in box = " << density << endl;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)
	
	inData >> outFile;
	outData.open(outFile,ios::out | ios::binary);
	cout << "File to record output: " << outFile << endl;
	inData >> dummyStr;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	inData >> processCLVs;
	cout << "Compute covariant vectors? " << processCLVs << endl;
	inData >> dummyStr;
	while (dummyStr.compare(str1) != 0){
		inData >> dummyStr;
	} // END while (dummyStr.compare(str1) != 0)

	if (processCLVs){
		inData >> clvSettleTime;
		cout << "Time to run to settle GS transients? " << clvSettleTime << endl;
		inData >> dummyStr;
		while (dummyStr.compare(str1) != 0){
			inData >> dummyStr;
		} // END while (dummyStr.compare(str1) != 0)
	}
	else {
		// IF NOT PROCESSING CLVs, SET THE SETTLE TIME TO SOMETHING VERY LARGE
		clvSettleTime = 1.0e9;
	}

	// CLOSE THE INPUT FILE
	inData.close();
	// END READ IN DATA FROM INPUT FILE AND OUTPUT VALUES TO SCREEN

	if (processCLVs){
		string   str2 (outFile);
		string   clvFile("clv_");
		clvFile.append(str2);
		clvData.open(clvFile,ios::out | ios::binary);
	} // END if (processCLVs)

	// SET PROGRAM CONSTANTS/DEFAULTS
	int   DIM    = 2;
	int phaseDim = 2*DIM*nDisks;
	int nlya     = phaseDim;
	float pi     = 4.0f*atan(1.0f);
	int modCLV   = 25;
	int lastPass = 0;
	int i_coll   = 0;

	// INITIALIZE COUNTERS AND TIME
	float Time = 0.f;

	// SET SIZE OF SIMULATION BOX
	// SOLVES N*pi*R^2 = density * A == density*(aspRatio*L^2) FOR L, DISKS HAVE RADIUS 0.5 BY CONSTRUCTION
	float boxSize = sqrt( ((float) nDisks) * pi / 4.0f / density );

	// Initialize the GPU (inherited from QR executable)
	initGPU(argc, argv);
	
	// CALL INITIALIZE ROUTINE TO POPULATE PHASE SPACE AND TANGENT VECTORS, ALONG WITH COLLISION TIMES
	cout << "mdgpu: initialize()" << endl;
	
	// declare pointers for allocation on the GPU
	// "ps" means phase-space: the physical states occupied by the reference trajectory (on the manifold)
	// "ts" means tangent-space: perturbations to the reference trajectory (lives in tangent space to the manifold at the reference point)
	float4 *d_ps, *d_ts;
	// pointers for the accumulator (cum) and the Lyapunov exponents (lyap)
	float *d_cum, *d_lyap;
	// array of structures to store collision times for all possible pairs (see utilitiesTypes.h for "Int2Float" declaration)
	Int2Float * d_fullct;
	// structure for next collision event; index, time and partner.
	Int2Float * d_cnext;
	
	// allocate memory on the GPU for pointers
	checkCudaErrors(cudaMalloc(&d_ps, sizeof(float4) * nDisks));
	checkCudaErrors(cudaMalloc(&d_ts, sizeof(float4) * nDisks * nlya));
	checkCudaErrors(cudaMalloc(&d_cum, sizeof(float) * nlya));
	checkCudaErrors(cudaMalloc(&d_fullct, sizeof(Int2Float)*nDisks*nDisks));
	checkCudaErrors(cudaMalloc(&d_cnext, sizeof(Int2Float)));
	checkCudaErrors(cudaMalloc(&d_lyap, sizeof(float)*nlya));
	
	// Initialization:
	// populate data in memory allocated to pointers
	initialize_gpu(d_ps,d_ts,d_fullct,d_cnext,d_cum,nDisks,nlya,DIM,boxSize);

	// allocate memory for data to be copied back from the GPU
	float4 *h_ps = (float4 *) malloc(sizeof(float4)*nDisks);
	//float4 *h_ts = (float4 *) malloc(sizeof(float4)*nDisks*nlya); // not in use, but possibly in the future.
	float *h_lyap= (float *) malloc(sizeof(float)*nlya);
	// END INITIALIZE
	
	/////////////////////////////////////////////////////
	//              MAIN LOOP                          //
	/////////////////////////////////////////////////////
	cout << "mdgpu: update()" << endl;
	for (i = 0; i <= nSteps; i++){

		// Perform QR factorization
		// Compute the Lyapunov exponents and renormalize the tangent vectors (to avoid exponential divergence)
		if ( (i % nOrtho) == 0 ){
			doQR(d_ts,nlya,nDisks,DIM,Time,d_cum,d_lyap);
		} // END if ( (i % nOrtho) == 0 )

		// Copy data from GPU and write data to file
		if ( (i % nOutput) == 0 ){
			checkCudaErrors(cudaMemcpy(h_ps,d_ps,sizeof(float4)*nDisks,cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(h_lyap,d_lyap,sizeof(float)*nlya,cudaMemcpyDeviceToHost));
			
			// OUTPUT CURRENT TIME AND LARGEST LYAPUNOV EXPONENT VALUES TO THE SCREEN AS A STATUS CHECK
			cout << "\t" << Time << "\t" << h_lyap[0] << endl;

			// OUTPUT CURRENT TIME TO FILE
			outData.write((char *) &Time, sizeof(float));

			// OUTPUT POSITION VECTOR AT CURRENT TIME TO FILE (SEQUENTIALLY BY VECTOR INDEX)
			for (j = 0; j < nDisks; j++){
				outData.write((char *) &h_ps[j].x, sizeof(float));
				outData.write((char *) &h_ps[j].y, sizeof(float));
				outData.write((char *) &h_ps[j].z, sizeof(float));
				outData.write((char *) &h_ps[j].w, sizeof(float));
			} // END for (j = 0; j < phaseDim; j++)

			
			// OUTPUT LYAPUNOV SPECTRUM AT CURRENT TIME
			for (j = 0; j < nlya; j++){
				outData.write((char *) &h_lyap[j], sizeof(float));
			} // END for (j = 0; j < nlya; ii++)

			// OUTPUT NUMBER OF COLLISIONS
			outData.write((char *) &i_coll, sizeof(int));

		} // END if ( (i % nOutput) == 0 )

		// Evolve state forward in time by "stepSize" units
		// This includes handling free flight and collision portions of the dynamics
		hardStep_gpu(d_ps,d_ts,d_fullct,d_cnext,nDisks,nlya,DIM,boxSize,stepSize,&i_coll);

		// INCREMENT TIME
		Time += stepSize;
	} // END for (i = 0; i < nSteps; i++)

	// FINALIZE (OUTPUT END-OF-RUN STATS AND PERFORM CLEANUP)
	cout << "mdgpu: finalize()" << endl;

	// OUTPUT MEAN FREE TIME (AVERAGE TIME BETWEEN SUCCESSIVE COLLISIONS FOR AN INDIVIDUAL DISK)
	cout << "Mean free time = " << (float)(nDisks)*Time / ((float)(i_coll)) << endl;

	// CLOSE MAIN OUTPUT FILE
	outData.close();

	// IF INDICATED FROM INPUT FILE, COMPUTE COVARIANT LYAPUNOV VECTORS
	// USING GINELLI'S METHOD (revisit this later)
	if (processCLVs){
		//computeCLVs();
	}

	// CLEAR ALLOCATED MEMORY
	delete [] outFile;
	nModes = 0;
	
	checkCudaErrors(cudaFree(d_ps));
	checkCudaErrors(cudaFree(d_ts));
	checkCudaErrors(cudaFree(d_cum));
	checkCudaErrors(cudaFree(d_lyap));
	checkCudaErrors(cudaFree(d_fullct));
	checkCudaErrors(cudaFree(d_cnext));

	free(h_ps);
	//free(h_ts);
	free(h_lyap);
	
	freeGPU();

	// PROGRAM RAN TO COMPLETION, EXIT NORMALLY
    return 0;
    
} // END int main (int argc, const char * argv[])

