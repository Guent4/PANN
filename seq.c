#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 10
#define M 11

int NUM_LAYERS = 3;
int *LAYER_SIZES;
float ETA = 0.5;

float **X;
float ***W;
float ***Z;
float *Y;

void printVector(float *vector, int len) {
	printf("------------------------------------\n");
	int i;
	for (i = 0; i < len; i++) {
		printf("%f\n", vector[i]);
	}
	printf("------------------------------------\n");
}

void printMatrix(float **matrix, int rows, int columns) {
    printf("------------------------------------\n");
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
    printf("------------------------------------\n");   
}

void printMatrixMatlab(float **matrix, int rows, int columns) {
    printf("----------------------------------------------------------------------\n");
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            printf("%f ", matrix[i][j]);
        }
        printf("; ");
    }
    printf("\n");
    printf("----------------------------------------------------------------------\n");
}

float increment(float val) {
    return val + 1;
}

float sigmoid(float val) {
    return (float)((double)1/(double)(1 + exp(-val)));
}

float sigmoidDeriv(float val) {
    return sigmoid(val) * (1 - sigmoid(val));
}

float sigmoidDerivWhenAlreadyHaveSigmoid(float val) {
    return val * (1 - val);
}

void createRandomMatrix(int rows, int cols, float ***mat) {
    int i;
    *mat = (float **)malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        (*mat)[i] = (float *)malloc(cols * sizeof(float));
    }
}

// Transpose a matrix
float** transpose(float **input, int num_rows, int num_cols) {
  int i, j;

  // Allocate the space for the new array
  float **matrix = (float **)calloc(num_cols, sizeof(float*));
  for (i = 0; i < num_cols; i++) {
    matrix[i] = (float *)calloc(num_rows, sizeof(float));
    for (j = 0; j < num_rows; j++) {
      matrix[i][j] = input[j][i];
    }
  }

  return matrix;
}

void matrixMatrixMultiply(float **A, int ARows, int ACols, float **B ,int BRows, int BCols, float ***C) {
    if (ACols != BRows) {
        printf("Dimension mismatch: %dx%d %dx%d\n", ARows, ACols, BRows, BCols);
        exit(1);
    }

    int i, j, k;
    
    *C = (float **)malloc(ARows * sizeof(float *));
    for (i = 0; i < ARows; i++) {
        (*C)[i] = (float *)malloc(BCols * sizeof(float));
    }

    for (i = 0; i < ARows; i++) {
        for (j = 0; j < BCols; j++) {
            float sum = 0.0;
            for (k = 0; k < ACols; k++) {
                sum += A[i][k] * B[k][j];
            }
            (*C)[i][j] = sum;
        }
    }
}

void matrixMatrixElementMultiply(float **A, float **B, int rows, int cols, float ***C) {
    int i, j;

    *C = (float **)malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        (*C)[i] = (float *)malloc(cols * sizeof(float));

        for (j = 0; j < cols; j++) {
            (*C)[i][j] = A[i][j] * B[i][j];
        }
    }
}

void matrixMatrixElementAdd(float **A, float **B, int rows, int cols, float ***C) {
    int i, j;

    *C = (float **)malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        (*C)[i] = (float *)malloc(cols * sizeof(float));
        
        for (j = 0; j < cols; j++) {
            (*C)[i][j] = A[i][j] + B[i][j];
        }
    }
}

void matrixElementApply(float **A, int rows, int cols, float(*f)(float)) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            A[i][j] = f(A[i][j]);
        }
    }
}

void matrixVectorSubtraction(float **matrix, int rows, float *vector, float ***result) {
    int i;

    *result = (float **)malloc(rows * sizeof(float **));
    for (i = 0; i < rows; i++) {
        (*result)[i] = (float *)malloc(sizeof(float));
        (*result)[i][0] = matrix[i][0] - vector[i];
    }
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

// Starting is included; ending is not
void readInXY(int starting, int ending) {
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
        // Only include interested N
        if (i >= starting && i < ending) {
            record = strtok(line, ",");
            
            // Put each token in the right location (X or Y)
            j = 0;
            while (record != NULL) {
                if (j == 0) {
                    Y[i] = atof(record);
                } else {
                    X[i][j-1] = atof(record);
                }

                j++;
                record = strtok(NULL, ",");
            }
        }

        i++;
    }

    // printVector(Y, N);
    // printMatrix(X, N, M);
    printMatrixMatlab(X, N, M);
}

void initializeMatrices() {
	int i, j;

	// Create input
	X = (float **)malloc(N * sizeof(float *));
	for (i = 0; i < N; i++) {
		X[i] = (float *)malloc(M * sizeof(float));
	}

	// Create output
	Y = (float *)malloc(N * sizeof(float));

    // Retrieve data from csv
    readInXY(0, 10);

    // Create weight matrices
    W = (float ***)malloc(NUM_LAYERS * sizeof(float **));
    for (i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? M : LAYER_SIZES[i-1];
        W[i] = (float **)malloc(numRows * sizeof(float *));
        for (j = 0; j < numRows; j++) {
            W[i][j] = (float *)calloc(LAYER_SIZES[i], sizeof(float));
        }

        matrixElementApply(W[i], numRows, LAYER_SIZES[i], increment);
    }

    // Create S matrices
    Z = (float ***)malloc((NUM_LAYERS - 1) * sizeof(float ***));
    for (i = 0; i < NUM_LAYERS - 1; i++) {
        Z[i] = (float **)malloc(N * sizeof(float *));
        for (j = 0; j < N; j++) {
            Z[i][j] = (float *)calloc(LAYER_SIZES[i], sizeof(float));
        }
    }
}

void freeMatrices() {
    int i, j;

    // Free X
    for (i = 0; i < N; i++) {
        free(X[i]);
    }
    free(X);

    // Free Y
    free(Y);

    // Free weights matrix
    for (i = 0; i < NUM_LAYERS; i++) {
        for (j = 0; j < M; j++) {
            free(W[i][j]);
        }
        free(W[i]);
    }
    free(W);

    // Free S matrix
    for (i = 0; i < NUM_LAYERS - 1; i++) {
        for (j = 0; j < N; j++) {
            free(Z[i][j]);
        }
        free(Z[i]);
    }
    free(Z);
}

void feedForward(float ***out) {
    int layer;

    float **in = X;
    int inRows = N;
    int inCols = M;

    printMatrix(X, N, M);
    for (layer = 0; layer < NUM_LAYERS; layer++) {
        // Multiply Z with W to get S
        int numRows = (layer == 0) ? M : (LAYER_SIZES[layer-1]);
        matrixMatrixMultiply(in, inRows, inCols, W[layer], numRows, LAYER_SIZES[layer], out);
        
        // Note that the output perceptrons do not have activation function
        if (layer == NUM_LAYERS - 1) break;

        // Apply activation function to S to get Z
        matrixElementApply(*out, inRows, LAYER_SIZES[layer], sigmoid);

        // Save Z because this is sigmoid(S) and is needed in back propagation
        Z[layer] = *out;
        
        // Update values for next iteration
        in = *out;
        inRows = inRows;
        inCols = LAYER_SIZES[layer];
    }
}

void backPropagation(float** estimation) {
    int layer;

    // Calculate the derivative of all S matrices
    for (layer = 0; layer < NUM_LAYERS - 1; layer++) {
        matrixElementApply(Z[layer], N, LAYER_SIZES[layer], sigmoidDerivWhenAlreadyHaveSigmoid);
    }

    // Get the error
    float **delta;
    matrixVectorSubtraction(estimation, N, Y, &delta);

    // Start propagating back to obtain deltas
    int deltaRows = N;
    int deltaCols = 1;
    int wRows = 1;
    int wCols = LAYER_SIZES[NUM_LAYERS - 2];

    float **transW;
    float **deltaW;
    for (layer = 0; layer < NUM_LAYERS - 1; layer++) {
        // Transpose W to multiply with delta
        transW = transpose(W[NUM_LAYERS - 1 - layer], wCols, wRows);
        matrixMatrixMultiply(delta, deltaRows, deltaCols, transW, wRows, wCols, &deltaW);

        // Element wise multiplication between deltaW and derivative of S
        matrixMatrixElementMultiply(deltaW, Z[NUM_LAYERS - 2 - layer], deltaRows, wCols, &delta);

        printMatrix(delta, deltaRows, wCols);

        // Free the temp arrays
        free(deltaW);
        free(transW);

        // Update matrix dimensions
        deltaRows = N;
        deltaCols = wCols;
        wRows = wCols;
        wCols = LAYER_SIZES[NUM_LAYERS - 3 - layer];
    }
}

int main(int argc, char** argv) {
    LAYER_SIZES = (int *)malloc(NUM_LAYERS * sizeof(int));
    LAYER_SIZES[0] = N;
    LAYER_SIZES[1] = 15;
    LAYER_SIZES[2] = 1;

	initializeMatrices();
    printVector(Y, N);

    float **out;
    feedForward(&out);
    backPropagation(out);

    freeMatrices();

    free(LAYER_SIZES);
}