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


#define BILLION 1000000000L
#define MILLION 1000000L
#define THOUSAND 1000L
#define TOTAL 8200

//ANN method
float testAccuracy(int testSize);
Matrix *feedForward(Matrix *in);
void backPropagation(Matrix *estimation);
void getXY(int starting, int ending, Matrix *inputs, Matrix *outputs);
void initializeMatrices(int testSize);
uint64_t get_dt(struct timespec *start, struct timespec *end);
void freeMatrices();

// cuda
__global__ void cuda_matirxElementSigmoid(float* A, int rows, int cols);
static cublasHandle_t handle;
static cublasStatus_t stat;
static cudaStream_t stream_hd0;
static cudaStream_t stream_hd1;
static cudaStream_t stream_dh0;


static int N;
static int FEATURES;
static int NUM_LAYERS;
static int *LAYER_SIZES;
static float ETA;
static float ERROR_THRESHOLD = 0.01;

static Matrix *XALL;
static Matrix *YALL;
static Matrix *XTS;
static Matrix *YTS;
static Matrix **WTS;
static Matrix **ZTS;

static Matrix *testX;
static Matrix *testY;

int main(int argc, char **argv) {

    // Set random seed
    srand(time(NULL));

    FEATURES = (argc > 1) ? strtol(argv[1], NULL, 10) : 5;
    N = (argc > 2) ? strtol(argv[2], NULL, 10) : 5;
    ETA = (argc > 3) ? atof(argv[3]) : 0.01;
    int testSize = (argc > 4) ? strtol(argv[4], NULL, 10) : 100;
    NUM_LAYERS = (argc > 5) ? strtol(argv[5], NULL, 10) : 3;

    printf("eta %f\n", ETA);

    // fill in the layer sizes
    LAYER_SIZES = (int *)malloc(NUM_LAYERS * sizeof(int));

    for (int i = 0; i < NUM_LAYERS; i++) {
        LAYER_SIZES[i] = (argc > 6+i) ? strtol(argv[6+i], NULL, 10) : 10;
    }
    LAYER_SIZES[NUM_LAYERS - 1] = 1; // This has to be 1

    initializeMatrices(testSize);

    // init cublas
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(1);
    }

    // create our streams
    cudaStreamCreate(&stream_hd0);
    cudaStreamCreate(&stream_hd1);
    cudaStreamCreate(&stream_dh0);

    struct timespec start, end; //timestamps
    struct timespec t_start, t_end; //timestamps
    uint64_t total_ff = 0;
    uint64_t total_bp = 0;
    uint64_t total_fd = 0;

    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &t_start);


    bool stop = false;
    for (int outer = 0; outer < 100 && !stop; outer++) {
        for (int iter = 0; iter < (TOTAL - testSize)/N && !stop; iter++) {
            // Retrieve data from csv
            clock_gettime(CLOCK_MONOTONIC, &start);
            getXY(iter*N, iter*N + N, XTS, YTS);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_fd += get_dt(&start, &end);

            clock_gettime(CLOCK_MONOTONIC, &start);
            Matrix *out = feedForward(XTS);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_ff += get_dt(&start, &end);

            clock_gettime(CLOCK_MONOTONIC, &start);
            backPropagation(out);
            clock_gettime(CLOCK_MONOTONIC, &end);
            total_bp += get_dt(&start, &end);

            stop = (testAccuracy(testSize) < ERROR_THRESHOLD);

            freeMatrix(out);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);


    float total_rt = get_dt(&t_start, &t_end);
    printf("RT: %f secs\n", total_rt/BILLION);
    float rt = (float)(total_bp + total_ff + total_fd);
    printf("Feed Forward: %f%%, Back prop %f%%, File Read: %f%%\n", 100*total_ff/rt, 100*total_bp/rt, 100*total_fd/rt);


    freeMatrices();

    free(LAYER_SIZES);
}

// starting is included; ending is not
void readXY()
{
    char buffer[2048];
    char *record, *line;
    int i, j;


    FILE* fstream = fopen("./dating/temp.csv", "r");

    if (fstream == NULL) {
        printf("\n file opening failed \n");
        exit(1);
    }

    i = -1;     // Starts at -1 to account for row of column headers
    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL) {
        if (i == -1) continue;

        record = strtok(line, ",");

        // Put each token in the right location (X or Y)
        j = 0;
        while (record != NULL) {
            if (j == 0) {
                XALL->m[IDXM(XALL, i, 0)] = atof(record);
            } else {
                YALL->m[IDXM(YALL, i, j-1)] = atof(record);
            }

            j++;
            record = strtok(NULL, ",");
        }

        i++;
    }

    fclose(fstream);
}


void getXY(int starting, int ending, Matrix *inputs, Matrix *outputs)
{
    inputs->m = &(XALL->m[IDXM(XALL, starting, 0)]);
    outputs->m = &(YALL->m[IDXM(YALL, starting, 0)]);

    inputs->rows = ending - starting;
    inputs->cols = XALL->cols;
    outputs->rows = ending - starting;
    outputs->cols = 1;
}


float testAccuracy(int testSize) {
    // Get the output
    Matrix *testOut = feedForward(testX);

    // Get the error
    Matrix *delta = matrixMatrixElementSub(testOut, testY);

    Matrix *trans = matrixTranspose(delta);
    freeMatrix(trans);

    float error = matrixReduceSumPow(delta, 2);
    printf("Error: %f\n", error);

    freeMatrix(delta);
    freeMatrix(testOut);

    return error;
}


Matrix *feedForward(Matrix *in)
{
    int wts_max = 0; //find max number of elements
    int max_cols = in->cols;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int tmp = WTS[layer]->cols*WTS[layer]->rows;
        wts_max = ( tmp > wts_max) ? tmp : wts_max;

        max_cols = (WTS[layer]->cols > max_cols) ? WTS[layer]->cols : max_cols;
    }

    float *dev_wts;
    float *dev_in;
    float *dev_z;
    float *dev_z_trans;
    cudaMalloc((void**)&dev_wts, wts_max*sizeof(float));
    cudaMalloc((void**)&dev_in, in->rows*max_cols*sizeof(float));
    cudaMalloc((void**)&dev_z, in->rows*max_cols*sizeof(float));
    cudaMalloc((void**)&dev_z_trans, in->rows*max_cols*sizeof(float));

    const float alpha = 1;
    const float beta = 0;


    const int in_rows = in->rows;
    int in_cols = in->cols;

    int wts_cols = WTS[0]->cols;
    int wts_rows = WTS[0]->rows;

    // this will load in transposed
    cublasSetMatrixAsync(in->cols, in->rows, sizeof(float),
            in->m, in->cols, dev_in, in->cols, stream_hd0);

    // set to our stream
    cublasSetStream(handle, stream_hd0);

    for (int layer = 0; layer < NUM_LAYERS; layer++) {

        wts_cols = WTS[layer]->cols;
        wts_rows = WTS[layer]->rows;

        if (in_cols != wts_rows) {
            printf("Error! Dimension mismatch in feedforward\n");
            exit(1);
        }

        // Load WTS[layer]  transposed
        cublasSetMatrixAsync(wts_cols, wts_rows, sizeof(float),
                WTS[layer]->m, wts_cols, dev_wts, wts_cols, stream_hd0);

        // multiply
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
            in_rows, wts_cols, in_cols, &alpha, dev_in, in_cols,
            dev_wts, wts_cols, &beta, dev_z, in_rows);

        // only apply sigmoid if not last layer
        if (layer == NUM_LAYERS - 1) // last output layer
            break;


        int blocks = UINT_DIV_CEIL((wts_cols*in_rows), BLOCK_SIZE);

        //printf("Launching kernel with dim y: %d, dim x: %d\n", UINT_DIV_CEIL(wts_cols, dimBlock.x), UINT_DIV_CEIL(in_rows , dimBlock.y));

        cuda_matirxElementSigmoid<<<blocks, BLOCK_SIZE, 0, stream_hd0>>>(dev_z, in_rows, wts_cols);


        cudaStreamSynchronize(stream_dh0); //Make sure the copy to host buffer is clear

        // stupid transpose to put it into col ordering
        // dev_z is dev_z[in_rows][wts_cols] (stored in row ordering on device)
        // now dev_z_trans is wts_cols x in_rows (stored in row ordering on device)
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, wts_cols, in_rows, &alpha,
                dev_z, in_rows, &beta, dev_z, wts_cols, dev_z_trans, wts_cols);

        // wait till done transposing so we can start async copy back to host
        cudaStreamSynchronize(stream_hd0);

        // now dev_z is WTS[layer]->cols x in->rows (stored in row ordering on device)
        cudaMemcpyAsync(ZTS[layer]->m, dev_z_trans,
            in_rows*wts_cols*sizeof(float), cudaMemcpyDeviceToHost, stream_dh0); //eventually make async

        //swap
        float *tmp = dev_in;
        dev_in = dev_z_trans;
        dev_z_trans = tmp;

        // update col dims
        in_cols = wts_cols;
    }

    Matrix *z = newMatrix(in_rows, WTS[NUM_LAYERS-1]->cols);

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, wts_cols, in_rows, &alpha, dev_z,
        in_rows, &beta, dev_z, wts_cols, dev_z_trans, wts_cols);

    cudaStreamSynchronize(stream_dh0);
    cudaStreamSynchronize(stream_hd0);

    // now dev_z is WTS[layer]->cols x in->rows (stored in row ordering on device)
    cudaMemcpy(z->m, dev_z_trans, in_rows*wts_cols*sizeof(float), cudaMemcpyDeviceToHost); //eventually make async


    cudaFree (dev_wts);
    cudaFree (dev_in);
    cudaFree (dev_z);
    cudaFree (dev_z_trans);

    // feed through last layer
    return z;
}


void backPropagation(Matrix *estimation) {
    // Backprop
    Matrix **D = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix *));

    for (int layer = NUM_LAYERS - 1; layer >= 0; layer--) {


        if (layer == NUM_LAYERS - 1) {
            Matrix *Dtrans = matrixMatrixElementSub(estimation, YTS);
            D[layer] = matrixTranspose(Dtrans);
            freeMatrix(Dtrans);
        } else {

            matrixElementApply(ZTS[layer], sigmoidDerivWhenAlreadyHaveSigmoid);
            Matrix *F = matrixTranspose(ZTS[layer]);

            Matrix *WD = matrixMatrixMultiply(WTS[layer + 1], D[layer + 1]);

            D[layer] = matrixMatrixElementMultiply(F, WD);

            freeMatrix(WD);
            freeMatrix(F);
        }
    }

    // Weight Updates
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        Matrix *DZ;
        if (layer == 0) {
            DZ = matrixMatrixMultiply(D[layer], XTS);
        } else {
            DZ = matrixMatrixMultiply(D[layer], ZTS[layer - 1]);
        }

        Matrix *wUpdates = matrixTranspose(DZ);
        float neta = -1*ETA;
        matrixElementApplyArg(wUpdates, multByConst, &neta);
        WTS[layer] = matrixMatrixElementAdd(WTS[layer], wUpdates);

        freeMatrix(wUpdates);
        freeMatrix(DZ);
    }

    // Free temporary matrices
    for (int i = 0; i < NUM_LAYERS; i++) {
        freeMatrix(D[i]);
    }
}


void initializeMatrices(int testSize) {
    // Get all data from csv
    XALL = newMatrix(TOTAL, FEATURES);
    YALL = newMatrix(TOTAL, 1);
    readXY();

	// Create input and output matrix for batch
    XTS = newMatrixSub(N, FEATURES);
    YTS = newMatrixSub(N, 1);

    // Create weight matrices
    WTS = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? FEATURES : LAYER_SIZES[i-1];

        WTS[i] = newMatrix(numRows, LAYER_SIZES[i]);

        // The in->firstHidden and lastHidden->out initially have weights of 0 and 1
        if (i == 0) {
            matrixElementApply(WTS[i], setTo0);
        } else if (i == NUM_LAYERS-1) {
            matrixElementApply(WTS[i], setTo1);
        } else {
            matrixElementApply(WTS[i], setToRand);
        }
    }

    // Create S matrices, must be page locked
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        ZTS[i] = newMatrixSub(N, LAYER_SIZES[i]);
        cudaMallocHost(&(ZTS[i]->m), ZTS[i]->rows*ZTS[i]->rows*sizeof(float));
    }

    // Get test data
    testX = newMatrixSub(testSize, FEATURES);
    testY = newMatrixSub(testSize, 1);

    // Retrieve test data from csv
    getXY(TOTAL-testSize, TOTAL, testX, testY);

}

void freeMatrices() {
    // Free X, Y
    freeMatrix(XALL);
    freeMatrix(YALL);

    freeMatrix(XTS);
    freeMatrix(YTS);

    // Free weights matrix
    for (int i = 0; i < NUM_LAYERS; i++) {
        freeMatrix(WTS[i]);
    }
    free(WTS);

    // Free Z matrix
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        cudaFreeHost(ZTS[i]->m);
        freeMatrix(ZTS[i]);
    }
    free(ZTS);

    freeMatrix(testY);
    freeMatrix(testX);
}




uint64_t get_dt(struct timespec *start, struct timespec *end)
{
    return BILLION*(end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec);
}



__global__ void cuda_matirxElementSigmoid(float* A, int rows, int cols)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;


    if (tid < rows*cols)
    {
        A[IDX2C(tid%rows, tid/rows, rows)] = 1.0/(1.0 + expf(-1*A[IDX2C(tid%rows, tid/rows, rows)]));
    }

}
