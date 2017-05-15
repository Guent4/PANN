#ifndef P_FEED_FORWARD
#define P_FEED_FORWARD

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "matrix.h"


#define UINT_DIV_CEIL(X,Y) (1 + (((X) - 1) / (Y)))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// 1d block size
#define BLOCK_SIZE 256


#ifdef __cplusplus
extern "C" {
#endif

// cuda
Matrix *feedForward(Matrix *in);
__global__ void cuda_matirxElementSigmoid(float* A, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif
