#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include "QR.hpp"
#include "device.h"

void sort_on_device(thrust::host_vector<float>& h_vec,thrust::host_vector<float>& h1_vec, int m, int n)
{
	// transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;
	thrust::device_vector<float> d1_vec = h1_vec;
 
    // note: d_vec.data() returns a device_ptr
 
    /*float* Q = thrust::raw_pointer_cast(d_vec.data());
	float* R = thrust::raw_pointer_cast(d1_vec.data());*/

	float* Q = new float [m * n];  
	float* R = new float [n * n];

	for (int i = 0; i < n * n; i++) R[i] = d1_vec[i];
	for (int i = 0; i < m * n; i++) Q[i] = d_vec[i];

	QR(Q,R,m,n);

	for (int i = 0; i < n * n; i++) d1_vec[i] = R[i];
	for (int i = 0; i < m * n; i++) d_vec[i] = Q[i];

	thrust::copy(d_vec.begin(),d_vec.end(),h_vec.begin());
	thrust::copy(d1_vec.begin(),d1_vec.end(),h1_vec.begin());

	// wrap raw pointer with a device_ptr 
 
    /*thrust::device_ptr<float> Q_ptr = thrust::device_pointer_cast(Q);
	thrust::device_ptr<float> R_ptr = thrust::device_pointer_cast(R);

	d_vec.data() = Q_ptr;
	d1_vec.data() = R_ptr;

	thrust::copy(d_vec.begin(),d_vec.end(),h_vec.begin());
	thrust::copy(d1_vec.begin(),d1_vec.end(),h1_vec.begin());*/

 
 

	/*
    // transfer data to the device
    thrust::device_vector<float> d_vec = h_vec;
	thrust::copy(d_vec.begin(), d_vec.end(), h1_vec.begin());

    // sort data on the device
    thrust::sort(d_vec.begin(), d_vec.end());
    
    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	*/
}

