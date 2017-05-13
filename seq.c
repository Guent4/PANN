
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
#include <unistd.h>

#include "matrix.h"


#define BILLION 1000000000L
#define MILLION 1000000L
#define THOUSAND 1000L
#define TOTAL 8200

//ANN method
float testAccuracy(int testSize);
Matrix *feedForward(Matrix *in);
void backPropagation(Matrix *estimation);
void getXY(int starting, int ending, Matrix *inputs, Matrix *outputs);
void initializeMatrices();
void freeMatrices();

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
    uint64_t diff;
    struct timespec start, end;

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
    
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    int stop = 0;
    for (int outer = 0; outer < 100 && stop == 0; outer++) {
        for (int iter = 0; iter < (TOTAL - testSize)/N && stop == 0; iter++) {
            // Retrieve data from csv
            getXY(iter*N, iter*N + N, XTS, YTS);

            Matrix *out = feedForward(XTS);
            backPropagation(out);

            float error = testAccuracy(testSize);
            stop = (error < ERROR_THRESHOLD) ? 1 : 0;

            freeMatrix(out);
        }
    }

    printf("Stopped %d\n", stop);

    // Stop timer
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate the time it took to perform calculation
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    diff = diff / MILLION;   // To get milliseconds from nanoseconds
    printf("elapsed time = %llu milliseconds\n", (long long unsigned int) diff);

    freeMatrices();

    free(LAYER_SIZES);
}

// starting is included; ending is not
void readXY() {
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

void getXY(int starting, int ending, Matrix *inputs, Matrix *outputs) {
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

Matrix *feedForward(Matrix *in) {
    Matrix *z = NULL;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        // Multiply Z with W to get S
        z = matrixMatrixMultiply(in, WTS[layer]);

        // Note that the output perceptrons do not have activation function
        if (layer == NUM_LAYERS - 1) break;

        // Apply activation function to S to get Z
        matrixElementApply(z, sigmoid);

        // Save Z because this is sigmoid(S) and is needed in back propagation
        ZTS[layer] = z;

        // Update values for next iteration
        in = z;
    }

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

    // Create S matrices
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        ZTS[i] = newMatrix(N, LAYER_SIZES[i]);
    }

    // Get test data
    testX = newMatrixSub(testSize, FEATURES);
    testY = newMatrixSub(testSize, 1);

    // Retrieve test data from csv
    getXY(TOTAL-testSize, TOTAL, testX, testY);

    printf("asdfsadfsadf\n");
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
        freeMatrix(ZTS[i]);
    }
    free(ZTS);

    freeMatrix(testY);
    freeMatrix(testX);
}
