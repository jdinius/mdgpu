############################################################################
# <summary> CMakeLists.txt for mdgpu. </summary>
# <date>    2014-08-03          </date>
# <author>  Joe Dinius </author>
# <email>   jdinius@math.arizona.edu </email>
############################################################################

cmake_minimum_required (VERSION 2.8)
project(mdgpu)
find_package(CUDA)
include(FindCUDA)

file( GLOB  hdr *.h )
file( GLOB  cu  *.cu )
file( GLOB  hpp *.hpp )

#add_definitions( -DTIMING )
add_definitions( -DDEBUG )
#add_definitions( -DMAGMA )
#add_definitions( -DLIST )

SET (mdgpu_files main.cpp)
set(CUDA_NVCC_FLAGS -arch=sm_21)
CUDA_ADD_EXECUTABLE(mdgpu ${mdgpu_files} ${hdr} ${hpp} ${cu})

target_link_libraries( mdgpu ${CUDA_cublas_LIBRARY} )