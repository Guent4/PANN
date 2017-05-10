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

#define TOTAL 8200


//ANN method
void testAccuracy(int testSize);
Matrix *feedForward(Matrix *in);
void backPropagation(Matrix *estimation);
void readInXY(int starting, int ending, Matrix *inputs, Matrix *outputs);
void initializeMatrices();
void freeMatrices();

void printVector(float *vector, int len);


static int N;
static int FEATURES;
static int NUM_LAYERS;
static int *LAYER_SIZES;
static float ETA = 0.005;


static Matrix *XTS;
static Matrix *YTS;
static Matrix **WTS;
static Matrix **ZTS;

int main(int argc, char **argv)
{
    //srand(time(NULL));
    srand(1);
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

    printf("test accuracy\n");
    testAccuracy(testSize);
    // printMatrix(WTS[1]);

    for (int outer = 0; outer < 100; outer++) {

        for (int iter = 0; iter < (TOTAL - testSize)/N; iter++) {
            // Retrieve data from csv
            readInXY(iter*N, iter*N + N, XTS, YTS);

            Matrix *out = feedForward(XTS);
            backPropagation(out);

            // printf("\n\n\n");
            if (iter % 20 == 0) {
                testAccuracy(testSize);
                // printMatrix(WTS[2]);
            }
            // printMatrix(WTS[1]);

            freeMatrix(out);
        }
    }


    freeMatrices();

    free(LAYER_SIZES);

}



// starting is included; ending is not
void readInXY(int starting, int ending, Matrix *inputs, Matrix *outputs)
{
    char buffer[2048];
    char *record, *line;
    int i, j;


    FILE* fstream = fopen("./dating/temp.csv", "r");

    if (fstream == NULL) {
        printf("\n file opening failed ");
        exit(1);
    }

    i = -1;     // Starts at -1 to account for row of column headers
    while((line = fgets(buffer, sizeof(buffer), fstream)) != NULL) {
        // Only include interested
        if (i >= starting && i < ending) {
            record = strtok(line, ",");

            // Put each token in the right location (X or Y)
            j = 0;
            while (record != NULL) {
                if (j == 0) {
                    outputs->m[IDXM(outputs, i-starting, 0)] = atof(record);
                } else {
                    inputs->m[IDXM(inputs, i-starting, j-1)] = atof(record);
                }

                j++;
                record = strtok(NULL, ",");
            }
        }

        i++;
    }
    fclose(fstream);

    // printMatrixMatlab(XTS);
}


void testAccuracy(int testSize)
{
    // Get test data
    Matrix *testX = newMatrix(testSize, FEATURES);
    Matrix *testY = newMatrix(testSize, 1);

    // Retrieve test data from csv
    readInXY(TOTAL-testSize, TOTAL, testX, testY);

    // Get the output
    Matrix *testOut = feedForward(testX);

    // Get the error
    Matrix *delta = matrixMatrixElementSub(testOut, testY);

    Matrix *trans = matrixTranspose(delta);
    printMatrix(trans);
    freeMatrix(trans);

    float error = matrixReduceSumPow(delta, 2);
    printf("Error: %f\n", error);

    freeMatrix(delta);
    freeMatrix(testOut);
    freeMatrix(testY);
    freeMatrix(testX);
}


Matrix *feedForward(Matrix *in)
{
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



void backPropagation(Matrix *estimation)
{

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



void initializeMatrices()
{

	// Create input
    XTS = newMatrix(N, FEATURES);

	// Create output
    YTS = newMatrix(N, 1);

    // Create weight matrices
    WTS = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? FEATURES : LAYER_SIZES[i-1];

        WTS[i] = newMatrix(numRows, LAYER_SIZES[i]);

        // The in->firstHidden and lastHidden->out have weights of 1
        if (i == 0) {
            matrixElementApply(WTS[i], setTo0);
        } else if (i == NUM_LAYERS-1) {
            matrixElementApply(WTS[i], setTo1);
            //WTS[i]->m[IDXM(WTS[i],0,0)] = 1;
        } else {
            matrixElementApply(WTS[i], setToRand);
        }

    }

    // Create S matrices
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (int i = 0; i < NUM_LAYERS - 1; i++) {
        ZTS[i] = newMatrix(N, LAYER_SIZES[i]);
    }
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
