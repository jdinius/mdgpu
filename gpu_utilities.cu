//
//  FILE:   gpu_utilities.cu
//  MODULE: mdgpu ("Molecular Dynamics for the GPU")
//
//  DESCRIPTION:
//  File contains all of the routines for running generalized collision simulations.
//
//  REFERENCE:
//  Dinius, J.  "Dynamical Properties of a Generalized Collision Rule for Multi-Particle Systems" Ph.D dissertation.
//
//  REVISION HISTORY:
//  Dinius, J.       Created                            09/07/14
//

// INCLUDES/DECLARATIONS
#include "utilities.h" // <see this file for global variable descriptions>
#include <iostream>
#include "stdlib.h"
#include <cmath>

#include "device_functions.h"
#include <cuda.h>
#include "QR.hpp"
#include "cublas.h"
//#include "device_launch_parameters.h"

using namespace std;

#define THREADS_PER_BLOCK_DIM 32
#define BIGTIME 10000000.0
#define NOCOLL  -1
#define MAXFLIGHT -2

// END INCLUDES/DECLARATIONS

void initialize_gpu(float4* ps,
	            float4* ts,
	            Int2Float* fullct,
				Int2Float* cnext,
				float* cum,
				int nDisks,
				int nlya,
				int DIM,
				float boxSize)
{

	// cuda kernel call setup (for nDisk-sized array calls)
	const int threadsPerBlock1D = 128;
	const int blocks1D = ceil((static_cast<float>(nDisks)/threadsPerBlock1D));
    const int blocks1D_nlya = ceil((static_cast<float>(nlya)/threadsPerBlock1D));
    
	int phaseDim = 2 * DIM * nDisks;

	// for 2D codes
	const dim3 blockSize(THREADS_PER_BLOCK_DIM,THREADS_PER_BLOCK_DIM);
	// for (nlya x phaseDim) calls
	const dim3 gridSize( ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(phaseDim)/THREADS_PER_BLOCK_DIM)) );
    // for (nDisks x nDisks) calls
	const dim3 gridSize_nd( ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) );
	// for (nlya x nDisks) calls
	dim3 gridSize_copy(ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) );
	
	size_t threadsPerBlock_nd = THREADS_PER_BLOCK_DIM * THREADS_PER_BLOCK_DIM;
	size_t blocksPerGrid_nd   = ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) * ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)); // total blocks in grid

	int n;   // round-off value of square root of nDisks calculation (check if nDisks is a perfect square).  This is used for initial configuration of disk positions.
    int sqrtN = (int)(sqrt((float)(nDisks)));        // square root of number of disks.  This is used for initial configuration of disk positions.
    float offSet = 0.0001243547654f; // small additive offset to ensure that initial positions are well-defined within the simulation box
    
	// INITIALIZE CLV STORAGE COUNTER (SAVE MEMORY)
	//countCLV = 0; // revisit this when addressing CLV computation- JWD 8-10-14

    // use kernel call to set y to all 0's for increased bandwidth allocation
	// see https://devtalk.nvidia.com/default/topic/394190/setting-arrays-to-a-value-float-arrays/
	float * d_y;
	checkCudaErrors(cudaMalloc(&d_y, sizeof(float)*phaseDim*nlya));
	init_to_x_kernel<<<gridSize,blockSize>>>(d_y, 0.0f, nlya, phaseDim);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// initialize the y vector to correspond to the identity matrix for initialization(put 1's in the correct places)
	put_1s_diag_kernel<<<gridSize,blockSize>>>(d_y,nlya,phaseDim);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	y_to_float4s_kernel<<<gridSize_copy,blockSize>>>(d_y,nlya,nDisks,DIM,ts);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	/*float4 *h_ts = (float4 *) malloc(sizeof(float4)*nlya * nDisks);
	checkCudaErrors(cudaMemcpy(h_ts,ts,sizeof(float4)*nlya*nDisks,cudaMemcpyDeviceToHost));*/

	// accumulator for lyapunov exponent calculations
	init_to_x_kernel<<<blocks1D_nlya,threadsPerBlock1D>>>(cum, 0.0f, nlya, 1);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// INITIALIZE PARAMETER n USED FOR DETERMINING INITIAL DISK LOCATIONS
    if (sqrtN*sqrtN == nDisks){
        // IF nDisks IS A PERFECT SQUARE, SET n EQUAL TO sqrtN
        n = sqrtN;
    } // END if (sqrtN*sqrtN == nDisks)
    
    else {
        // OTHERWISE, SET n TO THE INTEGER AFTER sqrtN
        n = sqrtN + 1;
    } // END else

	// SET DISK SPACING WIDTH IN X (dx) AND Y (dy)
	float dx = boxSize / ((float) n);
	//float dy = dx; // square box

	// put disks on the lattice
	initial_pos_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,nDisks,n,boxSize,dx,offSet);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
    // ENFORCE PERIODIC BOUNDARY CONDITIONS
    boxSet_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,nDisks,boxSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
	float* d_velocities;
	checkCudaErrors(cudaMalloc(&d_velocities, sizeof(float)*2*nDisks));
	unsigned long iSeed = 112324;

#ifdef RAND_ON_HOST
	// initialize random number generator with desired seed (use thrust random to take care of this, I think)
	float* h_velocities = (float *) malloc(sizeof(float)*2*nDisks);
	
	default_random_engine generator( iSeed );
	normal_distribution<float> distribution(0.0,1.0);
	
    for (unsigned int i = 0; i < DIM*nDisks; i++){
		// DRAW INITIAL MOMENTA FOR EACH DISK FROM A NORMAL DISTRIBUTION OF MEAN 0 AND STANDARD DEVIATION EQUAL TO SQRT(TEMPERATURE). TEMPERATURE IS HELD CONSTANT (AT 1) THROUGHOUT THE SIMULATION.
		h_velocities[i] = distribution(generator);
    } // END for (i = 0; i < 2*nDisks; i++)
    
	// copy to device memory
	checkCudaErrors(cudaMemcpy(d_velocities, h_velocities, sizeof(float) * 2 * nDisks, cudaMemcpyHostToDevice));
	free(h_velocities);
#else
	curandState* d_curandState;
	checkCudaErrors(cudaMalloc(&d_curandState, sizeof(curandState)*2*nDisks));
	init_curand_kernel<<<2*blocks1D,threadsPerBlock1D>>>(d_curandState,iSeed,2*nDisks);
	generate_normal_kernel<<<2*blocks1D,threadsPerBlock1D>>>(d_curandState,d_velocities,2*nDisks);
	checkCudaErrors(cudaFree(d_curandState));
#endif
	// convert into float4 type
	copy_velocities_kernel<<<blocks1D,threadsPerBlock1D>>>(d_velocities,ps,nDisks,DIM);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// compute the mean (third argument in chevrons is for shared memory allocation)
    float4 * d_interm; // first stage reduction allocation (used for both mean and KE reductions)
	float4 * d_bias; // second (last) reduction allocation (just a singleton)
	checkCudaErrors(cudaMalloc((void **) &d_interm, sizeof(float4)*blocks1D));
	checkCudaErrors(cudaMalloc((void **) &d_bias, sizeof(float4)));

	// first reduction (to block-sum)
	shmem_reduce_mean_kernel<<<blocks1D,threadsPerBlock1D,sizeof(float)*2*threadsPerBlock1D>>>(ps,d_interm,threadsPerBlock1D,nDisks,false);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// final reduction (sum over blocks = total sum)
	shmem_reduce_mean_kernel<<<1,blocks1D,sizeof(float)*2*blocks1D>>>(d_interm,d_bias,blocks1D,nDisks,true);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// remove mean bias (make mean 0 in each direction)
	remove_bias_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,d_bias,nDisks);
    
	// find total ke (= 1/2*sum_{i=1}^nDisks v_x^2+v_y^2)
	float * d_ke, * d_intflt;
	checkCudaErrors(cudaMalloc((void **) &d_ke, sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_intflt, sizeof(float)*blocks1D));

	// first reduction (to block-sum)
	shmem_reduce_sumsq_kernel<<<blocks1D,threadsPerBlock1D,sizeof(float)*threadsPerBlock1D>>>(ps,d_intflt,nDisks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// final reduction (sum over blocks = total sum)
	shmem_reduce_sumsq_kernel2<<<1,blocks1D,sizeof(float)*blocks1D>>>(d_intflt,d_ke,blocks1D);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// rescale to desired KE (= nDisks) so that avg. ke per disk = 1
	rescale_ke_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,d_ke,nDisks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// now, for the collision part!
	initialize_coll_times_kernel<<<gridSize_nd,blockSize>>>(fullct, nDisks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	compute_coll_times_kernel<<<gridSize_nd,blockSize>>>(ps, fullct, NOCOLL, NOCOLL, false, nDisks, 0.f, boxSize, 1.f);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	Int2Float * d_ctint; // first stage reduction allocation
	checkCudaErrors(cudaMalloc((void **) &d_ctint, sizeof(Int2Float)*blocksPerGrid_nd));
	
	// reduction (over blocks)
	shmem_min_kernel<<<gridSize_nd,blockSize,sizeof(Int2Float)*threadsPerBlock_nd>>>(fullct,d_ctint,nDisks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// final reduction (overall)
	shmem_min2_kernel<<<1,blocksPerGrid_nd,sizeof(Int2Float)*blocksPerGrid_nd>>>(d_ctint,cnext,blocksPerGrid_nd);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// clean-up local defs
	checkCudaErrors(cudaFree(d_y));
	checkCudaErrors(cudaFree(d_velocities));
	checkCudaErrors(cudaFree(d_interm));
	checkCudaErrors(cudaFree(d_bias));
    checkCudaErrors(cudaFree(d_ke));
	checkCudaErrors(cudaFree(d_intflt));
	checkCudaErrors(cudaFree(d_ctint));

	//
	
	return;
}

// convert tangent space vectors from single vector back to by-element
__global__ void y_to_float4s_kernel(float* const y,
	                                int nCols, // columns (nlya)
									int nRows, // rows (nDisks)
									int DIM,
									float4* tselem)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ( col >= nCols ||
		 row >= nRows )
	{
		return;
	}
	
	// get the 1D thread position:
	int thread_1d_pos = col*nRows+row; // position into tselem array of structs
	int y_1d_pos = 2*DIM*col*nRows+DIM*row; // position into y array
	tselem[thread_1d_pos].x = y[y_1d_pos+0]; //dx
	tselem[thread_1d_pos].y = y[y_1d_pos+1]; //dy
	tselem[thread_1d_pos].z = y[y_1d_pos+DIM*nRows]; //dvx // ensure same staggering as in mdcpu-serial
	tselem[thread_1d_pos].w = y[y_1d_pos+DIM*nRows+1]; //dvy
}

__global__ void float4s_to_y_kernel(float4* const tselem,
	                                int nCols,
							        int nRows,
									int DIM,
							        float* y)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ( col >= nCols ||
		 row >= nRows )
	{
		return;
	}
	
	// get the 1D thread position:
	int thread_1d_pos = col*nRows + row;
	int y_1d_pos = 2*DIM*col*nRows+DIM*row; // position into y array
	y[y_1d_pos+0]           = tselem[thread_1d_pos].x; // dx
	y[y_1d_pos+1]           = tselem[thread_1d_pos].y; // dy
	y[y_1d_pos+DIM*nRows]   = tselem[thread_1d_pos].z; // dvx
	y[y_1d_pos+DIM*nRows+1] = tselem[thread_1d_pos].w; // dvy

}

// doQR computes Lyapunov exponents and renormalizes tangent vectors
void doQR( float4* ts,
	       int nlya,
		   int nDisks,
		   int DIM,
		   float time,
		   float* cum,
		   float* lyap )
{
	// for QR
	float * d_Q, * d_R;
	int phaseDim = 2*DIM*nDisks;
	
	CUBLAS_SAFE_CALL(cublasAlloc (phaseDim * nlya, sizeof(*d_Q), (void**) &d_Q));
	CUBLAS_SAFE_CALL(cublasAlloc (nlya * nlya, sizeof(*d_R), (void**) &d_R));

	// for 2D codes
	const dim3 blockSize(THREADS_PER_BLOCK_DIM,THREADS_PER_BLOCK_DIM);
	// for (phaseDim x nlya) calls
	const dim3 gridSize( ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) );
    // for (nlya x nlya) calls
	const dim3 gridSize_nlya( ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)) );
    
	// for 1D codes
	const int threadsPerBlock1D = 128;
	const int blocks1D = ceil((static_cast<float>(nlya)/threadsPerBlock1D));

	// convert to float from ts array of structures (AoS)
	float4s_to_y_kernel<<<gridSize,blockSize>>>(ts,nlya,nDisks,DIM,d_Q);
	// initialize R matrix (to zeros)
	init_to_x_kernel<<<gridSize_nlya,blockSize>>>(d_R,0.f,nlya,nlya);
	
	// REORTHONORMALIZE TANGENT VECTORS
#ifdef MAGMA // future capability to pursue
#else
	QR(d_Q,d_R,phaseDim,nlya);
#endif

	// update accumulator and take the time-average
	lyapunov_kernel<<<blocks1D,threadsPerBlock1D>>>(d_R,cum,lyap,time,nlya);
	// copy back to ts AoS
	y_to_float4s_kernel<<<gridSize,blockSize>>>(d_Q,nlya,nDisks,DIM,ts);

	// clean up local allocations
	CUBLAS_SAFE_CALL(cublasFree(d_Q));
	CUBLAS_SAFE_CALL(cublasFree(d_R));

}

void hardStep_gpu(float4* ps,
	              float4* tselem,
				  Int2Float* ct,
				  Int2Float* cnext,
				  int nDisks,
				  int nlya,
				  int DIM,
				  float boxSize,
				  float dt_step,
				  int *i_coll)
{
	const int threadsPerBlock1D = 128;
	const int blocks1D = ceil((static_cast<float>(nDisks)/threadsPerBlock1D));

	const int blocks1D_nlya = ceil((static_cast<float>(nlya)/threadsPerBlock1D));

	const dim3 blockSize(THREADS_PER_BLOCK_DIM,THREADS_PER_BLOCK_DIM);
	const dim3 gridSize( ceil((static_cast<float>(nlya)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) );
  
	const dim3 gridSize_nd( ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)), ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) );

	size_t threadsPerBlock_nd = THREADS_PER_BLOCK_DIM * THREADS_PER_BLOCK_DIM;
	size_t blocksPerGrid_nd = ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)) * ceil((static_cast<float>(nDisks)/THREADS_PER_BLOCK_DIM)); // total blocks in grid

	Int2Float * d_ctint; // first stage reduction allocation
	checkCudaErrors(cudaMalloc((void **) &d_ctint, sizeof(Int2Float)*blocksPerGrid_nd));

	int * i_nextcoll;
	int * p_nextcoll;
	float * c_nextcoll;
	checkCudaErrors(cudaMalloc((void **) &i_nextcoll, sizeof(int)));
	checkCudaErrors(cudaMalloc((void **) &p_nextcoll, sizeof(int)));
	checkCudaErrors(cudaMalloc((void **) &c_nextcoll, sizeof(float)));

	int *d_bcount;
	checkCudaErrors(cudaMalloc((void **) &d_bcount, sizeof(int)));
	
	float *h_c  = (float *) malloc(sizeof(float));
	int   *h_i  = (int *) malloc(sizeof(int));
	int   *h_p  = (int *) malloc(sizeof(int));
	
	while (dt_step > 0.f){
		
		convert_from_int2float_kernel<<<1,1>>>( c_nextcoll, p_nextcoll, i_nextcoll, cnext, 1);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		checkCudaErrors(cudaMemcpy(h_c,c_nextcoll,sizeof(float),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_i,i_nextcoll,sizeof(int),cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_p,p_nextcoll,sizeof(int),cudaMemcpyDeviceToHost));
		
		if (dt_step < *h_c){			
			freeFlight_kernel<<<gridSize,blockSize>>>(ps,tselem,dt_step,nlya,nDisks,DIM);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
            // ENFORCE PERIODIC BOUNDARY CONDITIONS
            boxSet_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,nDisks,boxSize);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
            // UPDATE COLLISION TIMES FOR ALL DISKS
            compute_coll_times_kernel<<<gridSize_nd,blockSize>>>(ps, ct, NOCOLL, NOCOLL, true, nDisks, dt_step, boxSize, 1.f);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// update next time
			compute_coll_times_kernel<<<1,1>>>(ps, cnext, NOCOLL, NOCOLL, true, 1, dt_step, boxSize, 1.f);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

            // SET t TO ZERO, INDICATING CURRENT CALL TO hardStep IS COMPLETE
            dt_step = 0.f;
		}
		else {
			freeFlight_kernel<<<gridSize,blockSize>>>(ps,tselem,*h_c,nlya,nDisks,DIM);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			boxSet_kernel<<<blocks1D,threadsPerBlock1D>>>(ps,nDisks,boxSize);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			dt_step -= *h_c;
			
			// do collision here
			checkCudaErrors(cudaMemset(d_bcount,0,sizeof(int)));
			collision_kernel<<<blocks1D_nlya,threadsPerBlock1D>>>(ps,tselem,*h_i,*h_p,nlya,nDisks,boxSize,d_bcount,blocks1D_nlya);
			*i_coll += 1; // increment collision counter

			// update collision times
			compute_coll_times_kernel<<<gridSize_nd,blockSize>>>(ps, ct, *h_i, *h_p, true, nDisks, *h_c, boxSize, 1.f);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
			// minimum time reduction (over blocks)
			shmem_min_kernel<<<gridSize_nd,blockSize,sizeof(Int2Float)*threadsPerBlock_nd>>>(ct,d_ctint,nDisks);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
			// minimum time final reduction (overall)
			shmem_min2_kernel<<<1,blocksPerGrid_nd,sizeof(Int2Float)*blocksPerGrid_nd>>>(d_ctint,cnext,blocksPerGrid_nd);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}
	
	}

	// clean up
	checkCudaErrors(cudaFree(d_bcount));
	checkCudaErrors(cudaFree(d_ctint));
	checkCudaErrors(cudaFree(i_nextcoll));
	checkCudaErrors(cudaFree(p_nextcoll));
	checkCudaErrors(cudaFree(c_nextcoll));
	
	free(h_c);
	free(h_i);
	free(h_p);

}


__global__ void initial_pos_kernel(float4* ps,
	                               int nDisks,
							       int n,
							       float boxSize,
							       float dx,
								   float offSet)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float dy = dx; // square box
	
	if ( tid >= nDisks ){
		return;
	}
	
	// SET Y-COMPONENT OF DISK i 
	//y[DIM*i+1] = dy / 2.0 + (i / n) * dy + offSet;
	ps[tid].y = dy / 2.0 + (tid / n) * dy + offSet;

	// SET X-COMPONENT OF DISK i
	if ( (tid / n) % 2 == 0){
		// FOR EVERY OTHER ROW, STAGGER THE DISKS BY dx / 2 (CRYSTAL LATTICE PACKING)
		//y[DIM*i] = (i % n) * dx + dx / 4.0 + offSet;
		ps[tid].x = (tid % n) * dx + dx / 4.0 + offSet;
	} // END if ( (i / n) % 2 == 0)
	else {
		//y[DIM*i] = dx / 2.0 + (i % n) * dx + dx / 4.0 + offSet;
		ps[tid].x = dx / 2.0 + (tid % n) * dx + dx / 4.0 + offSet;
	} // END else

}
__global__ void boxSet_kernel(float4* ps,
	                          int nDisks,
							  float boxSize)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid >= nDisks ){
		return;
	}

	// FOR ALL DISKS
	while ( ps[tid].x < 0.f ){
		ps[tid].x += boxSize;
	}
	while ( ps[tid].x > boxSize ){
		ps[tid].x -= boxSize;
	}
	while ( ps[tid].y < 0.f ){
		ps[tid].y += boxSize;
	}
	while ( ps[tid].y > boxSize ){
		ps[tid].y -= boxSize;
	}

}

__global__ void init_to_x_kernel(float* y,
	                             float x,
	                             int nCols,
								 int nRows)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( c >= nCols ||
		 r >= nRows )
	{
		return;
	}

	int tid;
	if (nRows != 1){
		tid = r*nCols + c;
	}
	else {
		tid = c;
	}
	y[tid]=x;
}

__global__ void init_partner_kernel(int* y,
	                                int nCols,
								    int nRows)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( c >= nCols ||
		 r >= nRows )
	{
		return;
	}

	int tid = r*nCols + c;

	y[tid]=c; // set to column (that's the partner)
}
__global__ void put_1s_diag_kernel(float* y,
	                               int nCols,
							       int nRows)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( c >= nCols ||
		 r >= nRows )
	{
		return;
	}

	int tid = r*nCols + c;

	if (c == r){
		y[tid] = 1.0f;
	}
}

__global__ void copy_velocities_kernel(float* const y_in,
	                                   float4* y_out,
									   int nDisks,
									   int DIM)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid >= nDisks ){
		return;
	}

	y_out[tid].z = y_in[DIM*tid];   // velocity x comp
	y_out[tid].w = y_in[DIM*tid+1]; // velocity y comp

}

// from cs344 udacity Unit 3 snippets (with mods)
// put the mean scaling on the second (and final) reduction level
__global__ void shmem_reduce_mean_kernel(float4 * d_in,
	                                     float4 * d_out,  
						                 int size_offset,
										 int nDisks,
										 bool final_reduction)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	//	WHEN CALLING THIS KERNEL, ALLOCATE TWO TIMES nDisks FOR SHARED MEMORY (vx and vy channels are needed)
    extern __shared__ float sdata[];
	
	int myId = threadIdx.x + blockDim.x * blockIdx.x;

    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid]             = (myId < nDisks) ? d_in[myId].z : 0.f;
	sdata[tid+size_offset] = (myId < nDisks) ? d_in[myId].w : 0.f;

    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid]             += sdata[tid + s];
			sdata[tid+size_offset] += sdata[tid + size_offset + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x].z = sdata[0];
		d_out[blockIdx.x].w = sdata[size_offset];
		if (final_reduction){
			// do the average calculation on final reduction
			float n = (float)nDisks;
			d_out[blockIdx.x].z /= n;
			d_out[blockIdx.x].w /= n;
		}
    }
}

__global__ void remove_bias_kernel(float4* ps,
	                               float4* bias,
	                               int nDisks)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if ( tid >= nDisks ){
		return;
	}

	ps[tid].z -= bias[0].z;
	ps[tid].w -= bias[0].w;
}

__global__ void shmem_reduce_sumsq_kernel(float4 * d_in,
	                                      float * d_out,
										  int nDisks)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float tdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;

	int tid  = threadIdx.x;

    // load shared mem from global mem
    tdata[tid]             = (myId < nDisks) ? 0.5f*(d_in[myId].z*d_in[myId].z + d_in[myId].w*d_in[myId].w) : 0.f;
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            tdata[tid] += tdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = tdata[0];
    }
}

__global__ void shmem_reduce_sumsq_kernel2(float * d_in,
	                                       float * d_out,
										   int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
	extern __shared__ float udata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;

    int tid  = threadIdx.x;

    // load shared mem from global mem
    udata[tid]             = (myId < size) ? d_in[myId] : 0.f;
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            udata[tid] += udata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = udata[0];
    }
}

__global__ void rescale_ke_kernel(float4* ps,
	                              float* ke,
	                              int nDisks)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float n = (float)nDisks;

	if ( tid >= nDisks ){
		return;
	}

	ps[tid].z *= sqrt(n/ke[0]);
	ps[tid].w *= sqrt(n/ke[0]);
}

__global__ void initialize_coll_times_kernel( Int2Float* ct,
										      int nDisks )
{
	int c = blockIdx.x * blockDim.x + threadIdx.x; // disk to compute collision time with
	int r = blockIdx.y * blockDim.y + threadIdx.y; // current disk
	
	if ( c >= nDisks ||
		 r >= nDisks )
	{
		return;
	}
	
	int tid = r*nDisks + c;
	
	ct[tid].c = BIGTIME;
	ct[tid].i = r;
	ct[tid].p = c;

}

__global__ void compute_coll_times_kernel( float4* ps,
	                                       Int2Float* ct,
										   int index,
										   int partner,
										   bool update,
										   int nDisks,
										   float dt,
										   float boxSize,
										   float diam )
{
	int c = blockIdx.x * blockDim.x + threadIdx.x; // disk to compute collision time with
	int r = blockIdx.y * blockDim.y + threadIdx.y; // current disk
	
	if ( c >= nDisks ||
		 r >= nDisks )
	{
		return;
	}
	
	int tid = r*nDisks + c;
	if ( (index == NOCOLL) && !update ){ // initialize
		if (c > r){
			ct[tid].c = bin_time_func(ps[r],ps[c],boxSize,1.f);
		}
		else {
			ct[tid].c = BIGTIME;
		}
		ct[tid].i = r;
		ct[tid].p = c;
	}
	else { // update
		if (c > r){
			ct[tid].c -= dt;
			__syncthreads();

			if ( (r == index)   || (c == index) ||
				 (r == partner) || (c == partner) ){
				ct[tid].c = bin_time_func(ps[r],ps[c],boxSize,diam);
			}
		}
		else if (nDisks == 1){ // catch for when launching single kernel
			ct[tid].c -= dt;
		}
	}
}

__device__ inline float image_func( float y1,
	                                float y2,
							        float boxSize )
{
	float ret = y1-y2;

	if (ret > boxSize/2.f) ret-=boxSize;
	else if (ret < -boxSize/2.f) ret+=boxSize;

	return ret;
}

__device__ inline float bin_time_func( float4 y1, 
	                                   float4 y2,
								       float boxSize,
								       float diam )
{
	float dx  = image_func(y1.x,y2.x,boxSize);
	float dy  = image_func(y1.y,y2.y,boxSize);
	float dvx = y1.z-y2.z;
	float dvy = y1.w-y2.w;

	float delta = dx*dvx+dy*dvy;
	if (delta > 0.f){ // disks are not moving towards each other, so record a large time and exit
		return BIGTIME;
	}
	
	float dqsq  = dx*dx + dy*dy;
	float dvsq  = dvx*dvx + dvy*dvy;
	float discr = delta*delta - dvsq*(dqsq - diam);

	if (discr < 0){ // invalid condition for possible collision
		return BIGTIME;
	}

	// if all conditions are ok, then return the solution of the quadratic
	return (-delta - sqrt(discr)) / dvsq;
}

__global__ void shmem_min_kernel( Int2Float* ct_in,
	                              Int2Float* ct_out,
								  int nDisks )
{
	// sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ Int2Float wdata[];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int c = bx * blockDim.x + tx;
	int r = by * blockDim.y + ty;

    int myId = r*nDisks + c; // 1D location of current thread(myId < size) ? d_in[myId] : 0.f;
	
    int tid  = blockDim.y*threadIdx.x + threadIdx.y; // threadid within block
	int bid  = gridDim.y*blockIdx.x + blockIdx.y; // blockid within grid
    // load shared mem from global mem
    wdata[tid].c = (myId < nDisks*nDisks) ? ct_in[myId].c : BIGTIME;
	wdata[tid].i = (myId < nDisks*nDisks) ? ct_in[myId].i : NOCOLL;
	wdata[tid].p = (myId < nDisks*nDisks) ? ct_in[myId].p : NOCOLL;
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
	unsigned int threadsPerBlock = THREADS_PER_BLOCK_DIM*THREADS_PER_BLOCK_DIM;
    for (unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            //wdata[tid] += wdata[tid + s];
			if (wdata[tid+s].c < wdata[tid].c){
				wdata[tid].c = wdata[tid+s].c;
				wdata[tid].i = wdata[tid+s].i;
				wdata[tid].p = wdata[tid+s].p;
			}
        }
        __syncthreads();        // make sure all ops at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        ct_out[bid].c = wdata[0].c;
		ct_out[bid].i = wdata[0].i;
		ct_out[bid].p = wdata[0].p;
    }
}

__global__  void shmem_min2_kernel( Int2Float* ct_in,
	                                Int2Float* ct_out,
						            int size)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ Int2Float ydata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    ydata[tid].c = (myId < size) ? ct_in[myId].c : BIGTIME;
	ydata[tid].i = (myId < size) ? ct_in[myId].i : NOCOLL;
	ydata[tid].p = (myId < size) ? ct_in[myId].p : NOCOLL;

    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
			if (ydata[tid+s].c < ydata[tid].c){
				ydata[tid].c = ydata[tid+s].c;
				ydata[tid].i = ydata[tid+s].i;
				ydata[tid].p = ydata[tid+s].p;
			}
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        ct_out[blockIdx.x].c = ydata[0].c;
		ct_out[blockIdx.x].i = ydata[0].i;
		ct_out[blockIdx.x].p = ydata[0].p;
    }
}

__global__ void convert_2_int2float_kernel(float* ct,
	                                       int* p,
										   int* i,
										   Int2Float* out,
										   int size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= size){
		return;
	}
	out[tid].i = i[tid];
	out[tid].p = p[tid];
	out[tid].c = ct[tid];
}

__global__ void convert_from_int2float_kernel(float* ct,
	                                          int* p,
											  int* i,
										      Int2Float* in,
										      int size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= size){
		return;
	}
	ct[tid] = in[tid].c;
	i[tid]  = in[tid].i;
	p[tid]  = in[tid].p;
}

__global__ void freeFlight_kernel( float4* ps,
	                               float4* ts,
								   float dt,
								   int nCols,
								   int nRows,
								   int DIM)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	if ( c >= nCols ||
		 r >= nRows )
	{
		return;
	}

	//int tid = r*nCols + c;
	int tid = c*nRows + r;

	// update ps
	if (c == 0){
		ps[r].x += ps[r].z*dt;
		ps[r].y += ps[r].w*dt;
	}
	ts[tid].x += ts[tid].z*dt;
	ts[tid].y += ts[tid].w*dt;
}

__global__ void updateTimes_kernel( float4* ps,
	                                Int2Float*  ct,
									int     i1,
									int     i2,
									float   dt,
									float boxSize,
									int   size )
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	
	float tmp;

	if ( c >= size ||
		 r >= size )
	{
		return;
	}
	
	if (c == 0){
		ct[r].c -= dt;
	}
	__syncthreads();

	if ( (c == 0) &&
		( (r == i1) || (ct[r].p == i1) ||
		   (r == i2) || (ct[r].p == i2) ) ){
		ct[r].c = BIGTIME;
		ct[r].p  = MAXFLIGHT;
	}
	__syncthreads();

	if ( (i1 != NOCOLL) && (i2 != NOCOLL) ){
		if ( ( (r == i1) || (ct[r].p == i1) ||
			   (r == i2) || (ct[r].p == i2) ) &&
		       (c != r) ){

		    tmp = bin_time_func(ps[r],ps[c],boxSize,1.f);
			if (tmp < ct[r].c){
				ct[r].c = tmp;
				ct[r].p  = c;
			}
		}
	}
}

__global__ void collision_kernel( float4* ps,
	                              float4* ts,
								  int     i1, // index
								  int     i2, // partner
								  int   nCols, //nlya
								  int   nRows,
								  float boxSize,
								  int*  blockCount,
								  int   nBlocks)
{
	__shared__ float xs[5];
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	float x,y,vx,vy,vq;//,s;
	float dx,dy,dvx,dvy,dqq,dvq,dtc,dqcx,dqcy,dqcv,voffx,voffy;

	if ( c >= nCols ){
		return;
	}
	// only need to load the data into shared memory once (per block)
	if (threadIdx.x == 0){
		xs[0] = image_func(ps[i2].x,ps[i1].x,boxSize); // x2-x1
		xs[1] = image_func(ps[i2].y,ps[i1].y,boxSize); // y2-y1
		xs[2] = ps[i2].z-ps[i1].z; //vx2-vx1
		xs[3] = ps[i2].w-ps[i1].w; //vy2-vy1
		xs[4] = (xs[0]*xs[2]+xs[1]*xs[3]); //(x*vx+y*vy)/diam^2
		atomicAdd(blockCount,1); // count number of blocks that have executed
	}
	__syncthreads();

	// load shared data
	x  = xs[0];
	y  = xs[1];
	vx = xs[2];
	vy = xs[3];
	vq = xs[4];

	// update only after tangent space dynamics have been updated
	
	int tid_i1 =  c*nRows + i1;
	int tid_i2 =  c*nRows + i2;

	dx     = ts[tid_i2].x - ts[tid_i1].x;
	dy     = ts[tid_i2].y - ts[tid_i1].y;
	dvx    = ts[tid_i2].z - ts[tid_i1].z;
	dvy    = ts[tid_i2].w - ts[tid_i1].w;
	dqq    = dx*x+dy*y;
	dvq    = dvx*x+dvy*y;
	dtc    = -dqq / vq;
	dqcx   = (dx + vx*dtc);
	dqcy   = (dy + vy*dtc);
	dqcv   = dqcx*vx + dqcy*vy;
	voffx  = (dvq+dqcv)*x + vq*dqcx;
	voffy  = (dvq+dqcv)*y + vq*dqcy;

	// dq part
	ts[tid_i1].x += x*dqq;
	ts[tid_i1].y += y*dqq;
	ts[tid_i2].x -= x*dqq;
	ts[tid_i2].y -= y*dqq;
	// dv part
	ts[tid_i1].z += voffx;
	ts[tid_i1].w += voffy;
	ts[tid_i2].z -= voffx;
	ts[tid_i2].w -= voffy;
	__syncthreads();

	// if all blocks (and all threads within that block) have completed, update the phase space quantities
	if ( (*blockCount == nBlocks) &&
		 (c == (*blockCount-1)*blockDim.x) ){ // only do once (on thread 0 of the final block)
	    ps[i1].z += vq*x;
		ps[i1].w += vq*y;
		ps[i2].z -= vq*x;
		ps[i2].w -= vq*y;
	}
}

__global__ void lyapunov_kernel( float* R,
	                             float* cum,
								 float* lyap,
								 float time,
								 int size )
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= size){
		return;
	}

	int pos_R = tid*size + tid; // position along diagonal of R matrix from QR factorization

	// update accumulator
	cum[tid] += logf(R[pos_R]);
	// do Lyapunov exponent calculation
	lyap[tid] = cum[tid] / time;
}

#ifndef RAND_ON_HOST
__global__ void init_curand_kernel(curandState *state, 
	                               unsigned int seed, 
							       int size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= size){
		return;
	}
	
	curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generate_normal_kernel(curandState *state,
	                                   float *normal,
									   int size)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid >= size){
		return;
	}

	curandState lstate = state[tid];
	float rnd = curand_normal(&lstate);
	state[tid] = lstate;
	normal[tid] = rnd;
}
#endif