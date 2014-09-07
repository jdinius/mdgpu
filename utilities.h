//
//  FILE:   utilities.h
//  MODULE: mdgpu
//
//  DESCRIPTION:
//  File contains function and kernel declarations for running hard disk collision simulations.
//
//  REVISION HISTORY:
//  Dinius, J.       Initial release                              09/07/14
//

// BEGIN INCLUDES/DEFINITIONS
#ifndef mdgpu_utilities_h
#define mdgpu_utilities_h

#include "utilitiesTypes.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include "vector_types.h" // needed for cuda stuff
#include "cuda_runtime_api.h" // cudaDeviceSynchronize()
#include "cuda_runtime.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

using namespace std;

// END INCLUDES/DEFINITIONS

/* BEGIN GLOBAL VARIABLE DECLARATIONS
   NOTE ON GLOBAL DECLARATIONS: The statement (variable made global for memory allocation convenience) is made to indicate a global variable that is declared for convenient memory allocation ONLY.  These variables need not be global, as they are not used in multiple functions, however they are made global to prevent the re-allocation of memory each time functions are called that use these variables */
/* END GLOBAL VARIABLE DECLARATIONS */

/* BEGIN FUNCTION DECLARATIONS */
// cuda stuff
void initialize_gpu(float4* ps,
	            float4* ts,
	            Int2Float* fullct,
				Int2Float* cnext,
				float* cum,
				int nDisks,
				int nlya,
				int DIM,
				float boxSize);

void hardStep_gpu(float4* ps,
	              float4* tselem,
				  Int2Float* fullct,
				  Int2Float* cnext,
				  int nDisks,
				  int nlya,
				  int DIM,
				  float boxSize,
				  float dt_step,
				  int *i_coll);

void doQR( float4* ts,
	       int nlya,
		   int nDisks,
		   int DIM,
		   float time,
		   float* cum,
		   float* lyap );

// kernels
__global__ 
void init_to_x_kernel(float* y,
                      float x,
	                  int nCols,
					  int nRows);

__global__ 
void init_partner_kernel(int* y,
	                     int nCols,
					     int nRows);

__global__ 
void y_to_float4s_kernel(float* const y,
	                     int nCols,
				         int nRows,
						 int DIM,
				         float4* tselem);

__global__ 
void float4s_to_y_kernel(float4* const tselem,
	                     int nCols,
				         int nRows,
						 int DIM,
				         float* y);

__global__ 
void initial_pos_kernel(float4* ps,
	                    int nDisks,
						int n,
						float boxSize,
						float dx,
						float offSet);

__global__
void boxSet_kernel(float4* ps,
	               int nDisks,
				   float boxSize);

__global__ 
void copy_velocities_kernel(float* const y_in,
                            float4* y_out,
							int nDisks,
							int DIM);

__global__ 
void shmem_reduce_mean_kernel(float4 * d_in,
	                          float4 * d_out,  
						      int size_offset,
							  int nDisks,
							  bool final_reduction);

__global__ 
void remove_bias_kernel(float4* ps,
	                    float4* bias,
	                    int nDisks);

__global__
void shmem_reduce_sumsq_kernel(float4 * d_in,
	                           float * d_out,
							   int nDisks);

__global__ 
void shmem_reduce_sumsq_kernel2(float * d_in,
	                            float * d_out,
								int size);

__global__ 
void rescale_ke_kernel(float4* ps,
	                   float* ke,
	                   int nDisks);

__global__ 
void put_1s_diag_kernel(float* y,
	                    int nCols,
				        int nRows);

__global__ 
void initialize_coll_times_kernel( Int2Float* ct,
								   int nDisks );

__global__ 
void compute_coll_times_kernel( float4* ps,
	                            Int2Float* ct,
								int index,
								int partner,
								bool update,
								int nDisks,
								float dt,
								float boxSize,
								float diam );

__device__ 
inline float image_func( float y1,
	                     float y2,
				         float boxSize );

__device__ 
inline float bin_time_func( float4 y1, 
	                        float4 y2,
					        float boxSize,
					        float diam );

/* END FUNCTION DECLARATIONS */

__global__ 
void shmem_min_kernel( Int2Float* ct_in,
	                   Int2Float* ct_out,
					   int nDisks );
__global__
void shmem_min2_kernel( Int2Float* ct_in,
	                    Int2Float* ct_out,
						int size);
__global__ 
void convert_2_int2float_kernel(float* ct,
	                            int* p,
								int* i,
								Int2Float* out,
								int size);

__global__ 
void convert_from_int2float_kernel(float* ct,
	                               int* p,
								   int* i,
								   Int2Float* in,
								   int size);

__global__ 
void freeFlight_kernel( float4* ps,
	                    float4* ts,
						float dt,
						int nCols,
						int nRows,
						int DIM);

__global__ 
void updateTimes_kernel( float4* ps,
	                     Int2Float*  ct,
						 int     i1,
						 int     i2,
						 float   dt,
						 float boxSize,
						 int   size );

__global__ 
void collision_kernel( float4* ps,
	                   float4* ts,
					   int     i1,
					   int     i2,
					   int   nCols,
					   int   nRows,
					   float boxSize,
					   int*  blockCount,
					   int   nBlocks );

__global__ 
void lyapunov_kernel( float* R,
	                  float* cum,
					  float* lyap,
					  float time,
					  int size );
#endif
