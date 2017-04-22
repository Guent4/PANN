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
float **X;
float ***W;
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
    float after = (float)((double)1/(double)(1 + exp(-val)));
    return after;
}

void createRandomMatrix(int rows, int cols, float ***mat) {
    int i;
    *mat = (float **)malloc(rows * sizeof(float *));
    for (i = 0; i < rows; i++) {
        (*mat)[i] = (float *)malloc(cols * sizeof(float));
    }
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

void matrixElementApply(float **A, int rows, int cols, float(*f)(float)) {
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            A[i][j] = f(A[i][j]);
        }
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
            W[i][j] = calloc(LAYER_SIZES[i], sizeof(float));
        }

        matrixElementApply(W[i], numRows, LAYER_SIZES[i], increment);
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

    // Free weights metrix
    for (i = 0; i < NUM_LAYERS; i++) {
        for (j = 0; j < M; j++) {
            free(W[i][j]);
        }
        free(W[i]);
    }
    free(W);
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
        
        // Apply activation function to S to get Z
        matrixElementApply(*out, inRows, LAYER_SIZES[layer], sigmoid);

        in = *out;
        inRows = inRows;
        inCols = LAYER_SIZES[layer];
    }
}

int main(int argc, char** argv) {
    LAYER_SIZES = (int *)malloc(NUM_LAYERS * sizeof(int));
    LAYER_SIZES[0] = N;
    LAYER_SIZES[1] = 15;
    LAYER_SIZES[2] = 1;

	initializeMatrices();

    float **out;
    feedForward(&out);

    freeMatrices();

    free(LAYER_SIZES);
}