// Compile:         gcc -Wall par.c -lm
// Run:             ./a.out <features> <N> <eta> <testSize> <num_layers> <layer1> <layer2> ...
// Note that regardless if what is put for the last layer, program will overwrite last layer to have size 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>

// i is row j is col
#define IDXM(M, i, j) (i*M->cols + j)

#define TOTAL 8200

typedef struct {
    int rows;
    int cols;
    float *m;
} Matrix;

void initializeMatrices();
void printVector(float *vector, int len);

// Matrix helpers
void printMatrix(Matrix *matrix);
void printMatrixMatlab(Matrix *matrix);
void matrixElementApply(Matrix *A, float(*f)(float));

void freeMatrix(Matrix *matrix);
void freeMatrices();

//matrix element apply methods
float setTo1(float val);
float setToRand(float val);

static int N;
static int FEATURES;
static int NUM_LAYERS;
static int *LAYER_SIZES;
static float ETA = 0.005;


static Matrix *XTS;
static Matrix *YTS;
static Matrix **WTS;
static Matrix **ZTS;

int main(int argc, char **argv) {
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

    initializeMatrices();
    /*
    testAccuracy(testSize);
    // printMatrix(WTS[1]);

    int iter;
    int maxIters = (TOTAL - testSize) / N;
    for (iter = 0; iter < maxIters; iter++) {
        // Retrieve data from csv
        readInXY(iter*N, iter*N + N, XTS, YTS);

        Matrix *out = (Matrix *)malloc(sizeof(Matrix));
        feedForward(XTS, out);
        backPropagation(out);

        // printf("\n\n\n");
        testAccuracy(testSize);
        // printMatrix(WTS[1]);

        freeMatrix(out);
    }
    */

    freeMatrices();

    free(LAYER_SIZES);

}



void initializeMatrices()
{

	// Create input
    XTS = (Matrix *)malloc(sizeof(Matrix));
    XTS->m = (float *)malloc(sizeof(float[N][FEATURES]));
    XTS->rows = N;
    XTS->cols = FEATURES;

	// Create output
    YTS = (Matrix *)malloc(sizeof(Matrix));
    YTS->m = (float *)malloc(sizeof(float[N][1]));
    YTS->rows = N;
    YTS->cols = 1;

    // Create weight matrices
    WTS = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? FEATURES : LAYER_SIZES[i-1];

        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = numRows;
        matrix->cols = LAYER_SIZES[i];
        matrix->m = (float *)malloc(sizeof(float [numRows][LAYER_SIZES[i]]));
        WTS[i] = matrix;

        // The in->firstHidden and lastHidden->out have weights of 1
        if (i == 0 || i == NUM_LAYERS-1) {
            matrixElementApply(WTS[i], setTo1);
        } else {
            matrixElementApply(WTS[i], setToRand);
        }
        // printf("WEIGHT %d\n", i);
        // printMatrixMatlab(matrix);
    }

    // Create S matrices
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = NUM_LAYERS - 1;
        matrix->cols = N;
        matrix->m = (float*)malloc(sizeof(float[N][LAYER_SIZES[i]]));
        ZTS[i] = matrix;
    }
}

void matrixElementApply(Matrix *A, float(*f)(float)) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->m[IDXM(A,i,j)] = f(A->m[IDXM(A,i,j)]);
        }
    }
}


float setTo1(float val)
{
    return 1;
}


float setToRand(float val)
{
    return (float)rand()/(RAND_MAX);
}

void freeMatrix(Matrix *matrix)
{
    free(matrix->m);
    free(matrix);
}

void freeMatrices()
{
    // Free X, Y
    freeMatrix(XTS);
    freeMatrix(YTS);

    // Free weights matrix
    for (int i = 0; i < NUM_LAYERS; i++) {
        freeMatrix(WTS[i]);
    }
    free(WTS);

    // Free Z matrix
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        freeMatrix(ZTS[i]);
    }
    free(ZTS);
}


void printVector(float *vector, int len)
{
	printf("------------------------------------\n");
	int i;
	for (i = 0; i < len; i++) {
		printf("%f\n", vector[i]);
	}
	printf("------------------------------------\n");
}

void printMatrix(Matrix *matrix)
{
    printf("------------------------------------\n");
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%f\t", matrix->m[IDXM(matrix, i, j)]);
        }
        printf("\n");
    }
    printf("------------------------------------\n");
}

void printMatrixMatlab(Matrix *matrix)
{
    printf("----------------------------------------------------------------------\n");
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->m[IDXM(matrix, i, j)]);
        }
        printf("; ");
    }
    printf("\n");
    printf("----------------------------------------------------------------------\n");
}
