/*
  Copyright (C) 2009-2012 Fraunhofer SCAI, Schloss Birlinghoven, 53754 Sankt Augustin, Germany;
  
  This file is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

#include <cstdlib>
#include <ctime>

#  define CUDA_SAFE_CALL(call) {                                             \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUBLAS_SAFE_CALL(call) {                                           \
    cublasStatus err = call;                                                 \
    if( CUBLAS_STATUS_SUCCESS != err) {                                      \
        fprintf(stderr, "CuBlass error in file '%s' in line %i : %d.\n",     \
                __FILE__, __LINE__, err );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }

#  define CUBLAS_CHECK_ERROR(k) {                                             \
    cublasStatus err = cublasGetError();                                      \
    if( CUBLAS_STATUS_SUCCESS != err) {                                      \
        fprintf(stderr, "CuBlass error in file '%s' in line %i : it = %d, code = %d.\n",     \
                __FILE__, __LINE__, k, err );                                   \
        exit(EXIT_FAILURE);                                                  \
    } }

    
extern "C" {
	void initGPU(int argc, const char **argv);
	int QR(float* Q, float* R, unsigned int m, unsigned int n);
        void freeGPU();
        //double timestamp();
		time_t timestamp();
}
