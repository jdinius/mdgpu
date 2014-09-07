#pragma once

#include <thrust/host_vector.h>

// function prototype
void sort_on_device(thrust::host_vector<float>&V,thrust::host_vector<float>&V1, int m, int n);

