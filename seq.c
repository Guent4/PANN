// Compile:         gcc seq.c -lm
// Run:             ./a.out <features> <N> <eta> <testSize> <num_layers> <layer1> <layer2> ...
// Note that regardless if what is put for the last layer, program will overwrite last layer to have size 1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;
    float **m;
} Matrix;

int TOTAL = 8300;
int N;
int FEATURES;
int NUM_LAYERS;
int *LAYER_SIZES;
float ETA = 0.005;
float THRESHOLD = 0.5;

Matrix *XTS;
Matrix *YTS;
Matrix **WTS;
Matrix **ZTS;

int DONE = 0;

void printVector(float *vector, int len) {
	printf("------------------------------------\n");
	int i;
	for (i = 0; i < len; i++) {
		printf("%f\n", vector[i]);
	}
	printf("------------------------------------\n");
}

void printMatrix(Matrix *matrix) {
    printf("------------------------------------\n");
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%f\t", matrix->m[i][j]);
        }
        printf("\n");
    }
    printf("------------------------------------\n");
}

void printMatrixMatlab(Matrix *matrix) {
    printf("----------------------------------------------------------------------\n");
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->cols; j++) {
            printf("%f ", matrix->m[i][j]);
        }
        printf("; ");
    }
    printf("\n");
    printf("----------------------------------------------------------------------\n");
}

float setTo0(float val) {
    return 0;
}

float setTo1(float val) {
    return 1;
}

float setToRand(float val) {
    return (float)rand()/(RAND_MAX);
}

float roundToBinary(float val) {
    return (val > THRESHOLD) ? 1 : 0;
}

float multiplyByEta(float val) {
    return -val * ETA;
}

float sigmoid(float val) {
    return (float)((float)1/((float)1 + exp(-val)));
}

float sigmoidDeriv(float val) {
    return sigmoid(val) * (1 - sigmoid(val));
}

float sigmoidDerivWhenAlreadyHaveSigmoid(float val) {
    return val * (1 - val);
}

void createRandomMatrix(int rows, int cols, Matrix *matrix) {
    int i;
    float **mat = (float **)malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        mat[i] = (float *)malloc(cols * sizeof(float));
    }

    matrix->m = mat;
    matrix->rows = rows;
    matrix->cols = cols;
}

// Transpose a matrix
void transpose(Matrix *in, Matrix *out) {
  int i, j;

  // Allocate the space for the new array
  float **matrix = (float **)calloc(in->cols, sizeof(float*));
  for (i = 0; i < in->cols; i++) {
    matrix[i] = (float *)calloc(in->rows, sizeof(float));
    for (j = 0; j < in->rows; j++) {
      matrix[i][j] = in->m[j][i];
    }
  }

  out->m = matrix;
  out->rows = in->cols;
  out->cols = in->rows;
}

void matrixMatrixMultiply(Matrix *A, Matrix *B, Matrix *C) {
    if (A->cols != B->rows) {
        printf("Dimension mismatch: %dx%d %dx%d - matrixMatrixMultiply\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j, k;

    // Malloc the matrix C
    C->rows = A->rows;
    C->cols = B->cols;
    C->m = (float **)malloc(C->rows * sizeof(float *));
    for (i = 0; i < A->rows; i++) {
        C->m[i] = (float *)malloc(C->cols * sizeof(float));
    }

    // Fill in values for C
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < B->cols; j++) {
            float sum = 0.0;
            for (k = 0; k < A->cols; k++) {
                sum += A->m[i][k] * B->m[k][j];
            }
            C->m[i][j] = sum;
        }
    }
}

void matrixElementApply(Matrix *A, float(*f)(float)) {
    int i, j;

    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            A->m[i][j] = f(A->m[i][j]);
        }
    }
}

void matrixMatrixElementAdd(Matrix *A, Matrix *B, Matrix *C) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementAdd\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float **matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            matrix[i][j] = A->m[i][j] + B->m[i][j];
        }
    }
    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

void matrixMatrixElementSub(Matrix *A, Matrix *B, Matrix *C) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float **matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            matrix[i][j] = A->m[i][j] - B->m[i][j];
        }
    }
    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

void matrixMatrixElementSubP(Matrix *A, Matrix *B, Matrix *C) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float **matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            printf("%f - %f = %f\n", A->m[i][j], B->m[i][j], A->m[i][j] - B->m[i][j]);
            matrix[i][j] = A->m[i][j] - B->m[i][j];
        }
    }
    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

void matrixMatrixElementDiff(Matrix *A, Matrix *B, Matrix *C) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float **matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            matrix[i][j] = (A->m[i][j] - B->m[i][j] > 0.000001) ? 1 : 0;
        }
    }
    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

void matrixMatrixElementMultiply(Matrix *A, Matrix *B, Matrix *C)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch: %dx%d %dx%d - matrixMatrixElementMultiply\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float** matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < A->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            // printf("%fx%f = %f\n", A->m[i][j], B->m[i][j], A->m[i][j]*B->m[i][j]);
            matrix[i][j] = A->m[i][j] * B->m[i][j];
        }
    }

    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

float matrixReduceSumPow(Matrix *A, int exponent) {
    float sum = 0.0;

    int i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            sum += pow(A->m[i][j], exponent);
        }
    }

    return sum;
}

float getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ";"); tok && *tok; tok = strtok(NULL, ";\n")) {
        if (!--num) {
            printf("%s\n", tok);
            return atof(tok);
        }
    }

    printf("There Cannot be any empty values in the ANN data\n");
    exit(1);
}

// starting is included; ending is not
void readInXY(int starting, int ending, Matrix *inputs, Matrix *outputs) {
    char buffer[2048];
    char *record, *line;
    int i, j;


    // FILE* fstream = fopen("./dating/CleanedAndNoramlizedData.csv", "r");
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
                    outputs->m[i-starting][0] = atof(record);
                } else {
                    inputs->m[i-starting][j-1] = atof(record);
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

void initializeMatrices() {
	int i, j;

	// Create input
    XTS = (Matrix *)malloc(sizeof(Matrix));
	XTS->m = (float **)malloc(N * sizeof(float *));
	for (i = 0; i < N; i++) {
		XTS->m[i] = (float *)malloc(FEATURES * sizeof(float));
	}
    XTS->rows = N;
    XTS->cols = FEATURES;

	// Create output
    YTS = (Matrix *)malloc(sizeof(Matrix));
	YTS->m = (float **)malloc(N * sizeof(float *));
    for (i = 0; i < N; i++) {
        YTS->m[i] = (float *)malloc(sizeof(float));
    }
    YTS->rows = N;
    YTS->cols = 1;

    // Create weight matrices
    WTS = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix **));
    for (i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? FEATURES : LAYER_SIZES[i-1];
        float **w = (float **)malloc(numRows * sizeof(float *));
        for (j = 0; j < numRows; j++) {
            w[j] = (float *)malloc(LAYER_SIZES[i]*sizeof(float));
        }

        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = numRows;
        matrix->cols = LAYER_SIZES[i];
        matrix->m = w;
        WTS[i] = matrix;

        // The in->firstHidden and lastHidden->out have weights of 1
        if (i == 0) {
            matrixElementApply(WTS[i], setTo0);
        } else if (i == NUM_LAYERS-1) {
            // matrixElementApply(WTS[i], setTo1);
            matrixElementApply(WTS[i], setTo1);
            WTS[i]->m[0][0] = 1;
        } else {
            matrixElementApply(WTS[i], setToRand);
        }
        // printf("WEIGHT %d\n", i);
        // printMatrixMatlab(matrix);
    }
    // printf("\n\n\n");

    // Create S matrices
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (i = 0; i < NUM_LAYERS - 1; i++) {
        float **z = (float **)malloc(N * sizeof(float *));
        for (j = 0; j < N; j++) {
            z[j] = (float *)calloc(LAYER_SIZES[i], sizeof(float));
        }

        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = LAYER_SIZES[i];
        matrix->cols = N;
        matrix->m = z;
        ZTS[i] = matrix;
    }
}

void freeMatrix(Matrix *matrix) {
    int i;
    for (i = 0; i < matrix->rows; i++) {
        free(matrix->m[i]);
    }
    free(matrix->m);
    free(matrix);
}

void freeMatrices() {
    int i, j;

    // Free X, Y
    freeMatrix(XTS);
    freeMatrix(YTS);

    // Free weights matrix
    for (i = 0; i < NUM_LAYERS; i++) {
        freeMatrix(WTS[i]);
    }
    free(WTS);

    // Free Z matrix
    for (i = 0; i < NUM_LAYERS - 1; i++) {
        freeMatrix(ZTS[i]);
    }
    free(ZTS);
}

void feedForward(Matrix *in, Matrix *out) {
    int layer;
    Matrix *z;

    for (layer = 0; layer < NUM_LAYERS; layer++) {
        // Multiply Z with W to get S
        z = (Matrix *)malloc(sizeof(Matrix));
        matrixMatrixMultiply(in, WTS[layer], z);
        if (DONE) {
            printMatrix(z);
        }

        // Note that the output perceptrons do not have activation function
        if (layer == NUM_LAYERS - 1) break;

        // Apply activation function to S to get Z
        matrixElementApply(z, sigmoid);

        // Save Z because this is sigmoid(S) and is needed in back propagation
        ZTS[layer] = z;

        // Update values for next iteration
        in = z;
    }

    // Copy the result of multiplying Z with W
    out->m = z->m;
    out->rows = z->rows;
    out->cols = z->cols;

    free(z);
}

// void backPropagation(Matrix *estimation) {
//     int layer, i;

//     // Get the error
//     Matrix *delta = (Matrix *)malloc(sizeof(Matrix));
//     matrixMatrixElementSub(estimation, YTS, delta);

//     // printMatrix(delta);

//     for (layer = NUM_LAYERS - 1; layer >= 0; layer--) {
//         // Transpose W to multiply with delta
//         Matrix *transW = (Matrix *)malloc(sizeof(Matrix));
//         transpose(WTS[layer + 1], transW);

//         Matrix *deltaW = (Matrix *)malloc(sizeof(Matrix));
//         matrixMatrixMultiply(delta, transW, deltaW);

//         // Element wise multiplication between deltaW and derivative of S
//         freeMatrix(delta);
//         Matrix *delta = (Matrix *)malloc(sizeof(Matrix));
//         matrixElementApply(ZTS[layer], sigmoidDerivWhenAlreadyHaveSigmoid);
//         matrixMatrixElementMultiply(deltaW, ZTS[layer], delta);

//         if (layer >= 0) {
//             // Calculate how much the weights need to be updated by

//             // First transpose Z
//             Matrix *transposedZ = (Matrix *)malloc(sizeof(Matrix));
//             if (layer == 0) {
//                 transpose(XTS, transposedZ);
//             } else {
//                 transpose(ZTS[layer - 1], transposedZ);
//             }

//             // Now multiply Z with D
//             Matrix *wUpdates = (Matrix *)malloc(sizeof(Matrix));
//             matrixMatrixMultiply(transposedZ, delta, wUpdates);

//             // Multiply by -eta
//             matrixElementApply(wUpdates, multiplyByEta);

//             // Calculated how much to update by.  Now apply to W to update
//             // printMatrix(WTS[layer]);
//             matrixMatrixElementAdd(WTS[layer], wUpdates, WTS[layer]);
//             // printMatrix(WTS[layer]);

//             // Free up temp matrices
//             freeMatrix(transposedZ);
//             freeMatrix(wUpdates);
//         }

//         // Free temp matrices
//         freeMatrix(transW);
//         freeMatrix(deltaW);

//         // printf("\n\n\n");
//     }

//     // Free temporary matrices
//     freeMatrix(delta);
// }

void backPropagation(Matrix *estimation) {
    int layer, i;

    // Backprop
    Matrix **D = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix *));
    for (layer = NUM_LAYERS - 1; layer >= 0; layer--) {

        if (layer == NUM_LAYERS - 1) {
            Matrix *DTemp = (Matrix *)malloc(sizeof(Matrix));
            Matrix *Dtrans = (Matrix *)malloc(sizeof(Matrix));
            matrixMatrixElementSub(estimation, YTS, Dtrans);
            transpose(Dtrans, DTemp);
            free(Dtrans);
            D[layer] = DTemp;
        } else {
            // for (i = 0; i < NUM_LAYERS; i++) {
            //     printf("D%d    %dx%d\n", i, D[i]->rows, D[i]->cols);
            //     printf("W%d    %dx%d\n", i, WTS[i]->rows, WTS[i]->cols);
            //     if (i != NUM_LAYERS-1) {
            //         printf("F%d    %dx%d\n", i, ZTS[i]->cols, ZTS[i]->rows);
            //     } else {
            //         printf("F%d    %dx%d\n", i, YTS->rows, YTS->cols);
            //     }
            // }

            matrixElementApply(ZTS[layer], sigmoidDerivWhenAlreadyHaveSigmoid);
            Matrix *F = (Matrix *)malloc(sizeof(Matrix));
            transpose(ZTS[layer], F);

            Matrix *WD = (Matrix *)malloc(sizeof(Matrix));
            matrixMatrixMultiply(WTS[layer + 1], D[layer + 1], WD);

            Matrix *DTemp = (Matrix *)malloc(sizeof(Matrix));
            matrixMatrixElementMultiply(F, WD, DTemp);
            D[layer] = DTemp;

            free(WD);
            free(F);
        }
    }

    // Weight Updates
    for (layer = 0; layer < NUM_LAYERS; layer++) {
        Matrix *DZ = (Matrix *)malloc(sizeof(Matrix));
        if (layer == 0) {
            matrixMatrixMultiply(D[layer], XTS, DZ);
        } else {
            matrixMatrixMultiply(D[layer], ZTS[layer - 1], DZ);
        }
        Matrix *wUpdates = (Matrix *)malloc(sizeof(Matrix));
        transpose(DZ, wUpdates);
        matrixElementApply(wUpdates, multiplyByEta);
        matrixMatrixElementAdd(WTS[layer], wUpdates, WTS[layer]);

        free(wUpdates);
        free(DZ);
    }

    // Free temporary matrices
    for (i = 0; i < NUM_LAYERS; i++) {
        freeMatrix(D[i]);
    }
}

void testAccuracy(int testSize) {
    int i, j;

    // Get test data
    Matrix *testX = (Matrix *)malloc(sizeof(Matrix));
    testX->m = (float **)malloc(testSize * sizeof(float *));
    for (i = 0; i < testSize; i++) {
        testX->m[i] = (float *)malloc(FEATURES * sizeof(float));
    }
    testX->rows = testSize;
    testX->cols = FEATURES;

    Matrix *testY = (Matrix *)malloc(sizeof(Matrix));
    testY->m = (float **)malloc(testSize * sizeof(float *));
    for (i = 0; i < testSize; i++) {
        testY->m[i] = (float *)malloc(sizeof(float));
    }
    testY->rows = testSize;
    testY->cols = 1;

    // Retrieve test data from csv
    readInXY(TOTAL-testSize, TOTAL, testX, testY);

    // Get the output
    Matrix *testOut = (Matrix *)malloc(sizeof(Matrix));
    feedForward(testX, testOut);

    // TESTING DATA OUTPUT
    Matrix *outTrans = (Matrix *)malloc(sizeof(Matrix));
    transpose(testOut, outTrans);
    // printMatrix(outTrans);
    free(outTrans);

    // Get the error
    Matrix *delta = (Matrix *)malloc(sizeof(Matrix));
    matrixMatrixElementSub(testOut, testY, delta);

    Matrix *trans = (Matrix *)malloc(sizeof(Matrix));
    transpose(delta, trans);
    printMatrix(trans);
    free(trans);

    float error = matrixReduceSumPow(delta, 2);
    printf("Error: %f\n", (float)error);

    // // Modify the output so that it's binary
    // matrixElementApply(testOut, roundToBinary);
    // matrixMatrixElementDiff(testOut, testY, delta);

    // float errorPerc = matrixReduceSumPow(delta, 1);
    // printf("Error: %f%%\n", (float)errorPerc*100 / (float)(testSize));

    freeMatrix(delta);
    freeMatrix(testOut);
    freeMatrix(testY);
    freeMatrix(testX);
}

int main(int argc, char** argv) {
    //srand(time(NULL));
    srand(1);
    FEATURES = (argc > 1) ? strtol(argv[1], NULL, 10) : 5;
    N = (argc > 2) ? strtol(argv[2], NULL, 10) : 5;
    ETA = (argc > 3) ? atof(argv[3]) : 0.01;
    int testSize = (argc > 4) ? strtol(argv[4], NULL, 10) : 100;
    NUM_LAYERS = (argc > 5) ? strtol(argv[5], NULL, 10) : 3;

    printf("eta %f\n", ETA);

    LAYER_SIZES = (int *)malloc(NUM_LAYERS * sizeof(int));

    int i;
    for (i = 0; i < NUM_LAYERS; i++) {
        LAYER_SIZES[i] = (argc > 6+i) ? strtol(argv[6+i], NULL, 10) : 10;
    }
    LAYER_SIZES[NUM_LAYERS - 1] = 1; // This has to be 1

    initializeMatrices();
    testAccuracy(testSize);
    // printMatrix(WTS[1]);

    int outer;
    for (outer = 0; outer < 100; outer++) {
        int iter;
        int maxIters = (TOTAL - testSize) / N;
        for (iter = 0; iter < maxIters; iter++) {
            // Retrieve data from csv
            readInXY(iter*N, iter*N + N, XTS, YTS);

            Matrix *out = (Matrix *)malloc(sizeof(Matrix));
            feedForward(XTS, out);
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

    // printf("\n\n\n");
    // for (i = 0; i < NUM_LAYERS; i++) {
    //     printMatrix(WTS[i]);
    // }

    freeMatrices();

    free(LAYER_SIZES);
}
