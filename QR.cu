/*
  Copyright (C) 2009-2012 Fraunhofer SCAI, Schloss Birlinghoven, 53754 Sankt Augustin, Germany;
  all rights reserved unless otherwise stated.
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston,
  MA 02111-1307 USA
*/

#include <cstdio>
//#include <sys/time.h>

#include "cublas.h"

#include "QR.hpp"
#include "GPUTimer.hpp"

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

int QR(float* QGPU, float* RGPU, unsigned int m, unsigned int n) 
{
  /*float* QGPU;
  float* RGPU;*/
  
  GPUTimer timeMult;
  GPUTimer timeScale;
  GPUTimer timeUpdate;

#if defined(TIMING) and defined(LIST)
  FILE* f;
  char filename[256];
#endif

  // already allocated, so don't do this

  /*printf("allocate QGPU (%d, %d)\n", m, n);

  CUBLAS_SAFE_CALL(cublasAlloc (m * n, sizeof(*QGPU), (void**) &QGPU));

  printf("allocate RGPU (%d, %d)\n", n, n);*/

  /*CUBLAS_SAFE_CALL(cublasAlloc (n * n, sizeof(*RGPU), (void**) &RGPU));

  CUBLAS_SAFE_CALL(cublasSetMatrix (m, n, sizeof(*QGPU), Q, m, QGPU, m));

  CUBLAS_SAFE_CALL(cublasSetMatrix (m, n, sizeof(*RGPU), R, n, RGPU, m));*/
      
#if defined(TIMING) and defined(LIST)
  sprintf(filename, "list.cublas.%d_%d", m, n);
  f = fopen(filename, "w");
#endif

  for (unsigned int k = 1; k <= n; k++) {

    // call sgemv('T', M, N-K+1, 1.0, Q(1,K), M, Q(1,K), 1, 0.0, R(K,K), N)

    timeMult.start();
    cublasSgemv('T', m, n-k+1, 1.0, &QGPU[IDX2F(1,k,m)], m, &QGPU[IDX2F(1,k,m)], 1, 0.0, &RGPU[IDX2F(k,k,n)], n);
    CUBLAS_CHECK_ERROR(k);
    timeMult.stop();

    float S;

    CUBLAS_SAFE_CALL(cublasGetMatrix(1, 1, sizeof(S), &RGPU[IDX2F(k,k,n)], n, &S, n));

    S = sqrt(S);

    // call sscal(m, 1.0 / S, Q(1,K), 1)

    timeScale.start();
    cublasSscal(m, 1.0f / S, &QGPU[IDX2F(1,k,m)], 1);
    CUBLAS_CHECK_ERROR(k);

    // call sscal(n-k+1, 1.0/ S, R(K,K), N)

    cublasSscal(n-k+1, 1.0f / S, &RGPU[IDX2F(k,k,n)], n);
    CUBLAS_CHECK_ERROR(k);
    timeScale.stop();

    // call sger(M, N-K, -1.0, Q(1,K), 1, R(K,K+1), N, Q(1,K+1), M)

    timeUpdate.start();
    cublasSger(m, n-k, -1.0, &QGPU[IDX2F(1,k,m)], 1, &RGPU[IDX2F(k,k+1,n)], n, &QGPU[IDX2F(1,k+1,m)], m);
    CUBLAS_CHECK_ERROR(k);
    timeUpdate.stop();

#if defined(TIMING) and defined(LIST)
    fprintf(f, "%d %f %f %f\n", n + 1 - k, timeMult.last(), timeScale.last(), timeUpdate.last());
#endif

  }
  
#if defined(TIMING) and defined(LIST)
  fclose(f);
#endif

  /*printf("Detailed timings: mult = %g, scale = %g, update = %g\n",
          timeMult.get(), timeScale.get(), timeUpdate.get());

  CUBLAS_SAFE_CALL(cublasGetMatrix (n, n, sizeof(*R), RGPU, n, R, n));
  CUBLAS_SAFE_CALL(cublasGetMatrix (m, n, sizeof(*Q), QGPU, m, Q, n));

  CUBLAS_SAFE_CALL(cublasFree(RGPU));
  CUBLAS_SAFE_CALL(cublasFree(QGPU));*/

  return EXIT_SUCCESS; 
}

/* ---------------------------------------------------------------------- */

/* Function to initialize the GPU. Takes the commandline arguments to allow to
   choose the GPU. */

void initGPU(int argc, const char **argv)
{
  int deviceCount;

  CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

#ifdef DEBUG
  printf ("initGPU: %d devices available\n", deviceCount);
#endif

  int device = 0;   // default device

#ifdef DEBUG
  printf ( "Number of arguments = %d\n", argc);
#endif

  if (argc > 1) {
     sscanf(argv[1], "%d", &device);
  }

#ifdef DEBUG
  printf ("initGPU: try to use device %d\n", device);
#endif

  cudaDeviceProp deviceProp;

  CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, device));

  if (deviceProp.major < 1) {
     fprintf(stderr, "cutil error: device %d does not support CUDA.\n", device);
     exit(-1);
  }

  fprintf(stderr, "Using device %d: %s\n", device, deviceProp.name);

  CUDA_SAFE_CALL(cudaSetDevice(device));

  CUBLAS_SAFE_CALL(cublasInit());
}

/* ---------------------------------------------------------------------- */

void freeGPU()
{
  CUBLAS_SAFE_CALL(cublasShutdown());
}
