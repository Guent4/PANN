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
#include <mpi.h>


#include "matrix.h"
#include "pfeed_forward.h"


#define BILLION 1000000000L
#define MILLION 1000000L
#define THOUSAND 1000L
#define TOTAL 8200
#define SEED 97


//ANN method
float testAccuracy(int testSize);
void backPropagation(Matrix *estimation);
void getXY(int starting, int ending, Matrix *inputs, Matrix *outputs);
void initializeMatrices(int testSize);
uint64_t get_dt(struct timespec *start, struct timespec *end);
void freeMatrices();
void mpi_avg_weights();

// public
cublasHandle_t handle;
//private
static cublasStatus_t stat;

//private
static int N;
static int FEATURES;
static int *LAYER_SIZES;
static float ETA;
static float ERROR_THRESHOLD = 0.01;
//public
int NUM_LAYERS;

// private
static Matrix *XALL;
static Matrix *YALL;
static Matrix *XTS;
static Matrix *YTS;

static Matrix *testX;
static Matrix *testY;

//for mpi
static int myrank, nprocs;

//public
Matrix **WTS;
Matrix **ZTS;


int main(int argc, char **argv) {

    // Set random seed
    srand(SEED);

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

    //init mpi
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int name_len;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Node '%s' (rank %d) online and reporting for duty!\n", processor_name, myrank);

    initializeMatrices(testSize);

    // init cublas
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(1);
    }

    struct timespec start, end; //timestamps
    struct timespec t_start, t_end; //timestamps
    uint64_t total_ff = 0;
    uint64_t total_bp = 0;
    uint64_t total_fd = 0;

    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int blocks_proc = ((TOTAL - testSize)/N)/nprocs;

    for (int outer = 0; outer < 10000; outer++) {
        for (int iter = blocks_proc*myrank; iter < blocks_proc*(myrank+1); iter++) {
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

            //stop = (testAccuracy(testSize) < ERROR_THRESHOLD);

            freeMatrix(out);
        }
        mpi_avg_weights();
        // see if done
        float err = testAccuracy(testSize);
        if (!myrank)
            printf("Error: %f\n", err);

        if (testAccuracy(testSize) < ERROR_THRESHOLD)
            break;

    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    if (!myrank) {
        float total_rt = get_dt(&t_start, &t_end);
        printf("RT: %f secs\n", total_rt/BILLION);
        float rt = (float)(total_bp + total_ff + total_fd);
        printf("Feed Forward: %f%%, Back prop %f%%, File Read: %f%%\n", 100*total_ff/rt, 100*total_bp/rt, 100*total_fd/rt);
    }

    freeMatrices();

    free(LAYER_SIZES);

    MPI_Finalize();
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

    freeMatrix(delta);
    freeMatrix(testOut);

    return error;
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
        cudaMallocHost((void**)(&(ZTS[i]->m)), ZTS[i]->rows*ZTS[i]->rows*sizeof(float));
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



void mpi_avg_weights()
{

    for (int i = 0; i < NUM_LAYERS; i++) {
        uint32_t num = (WTS[i]->cols)*(WTS[i]->rows);
        float *recv = malloc(num*sizeof(float));
        // send results
        MPI_Allreduce( WTS[i]->m, recv, num,
        MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        // update weights
        memcpy(WTS[i]->m, recv, num*sizeof(float));
        //avg
        float mult = 1.0/nprocs;
        matrixElementApplyArg(WTS[i], multByConst, &mult);
    }

}
