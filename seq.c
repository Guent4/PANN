#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N 10
#define M 11

typedef struct {
    int rows;
    int cols;
    float **m;
} Matrix;

int NUM_LAYERS = 3;
int *LAYER_SIZES;
float ETA = 0.5;

Matrix *XTS;
Matrix *YTS;
Matrix **WTS;
Matrix **ZTS;

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

float increment(float val) {
    return val + 1;
}

float multiplyByEta(float val) {
    return -val * ETA;
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

void matrixMatrixElementMultiply(Matrix *A, Matrix *B, Matrix *C) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch: %dx%d %dx%d - matrixMatrixElementMultiply\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    float** matrix = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < A->rows; i++) {
        matrix[i] = (float *)malloc(A->cols * sizeof(float));

        for (j = 0; j < A->cols; j++) {
            matrix[i][j] = A->m[i][j] * B->m[i][j];
        }
    }

    C->m = matrix;
    C->rows = A->rows;
    C->cols = A->cols;
}

void matrixMatrixElementAdd(Matrix *A, Matrix *B, Matrix *matrix) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementAdd\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    matrix->m = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix->m[i] = (float *)malloc(A->cols * sizeof(float));
        
        for (j = 0; j < A->cols; j++) {
            matrix->m[i][j] = A->m[i][j] + B->m[i][j];
        }
    }
    matrix->rows = A->rows;
    matrix->cols = A->cols;
}

void matrixElementApply(Matrix *A, float(*f)(float)) {
    int i, j;

    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->cols; j++) {
            A->m[i][j] = f(A->m[i][j]);
        }
    }
}

void matrixMatrixElementSub(Matrix *A, Matrix *B, Matrix *matrix) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    int i, j;

    matrix->m = (float **)malloc(A->rows * sizeof(float *));
    for (i = 0; i < B->rows; i++) {
        matrix->m[i] = (float *)malloc(A->cols * sizeof(float));
        
        for (j = 0; j < A->cols; j++) {
            matrix->m[i][j] = A->m[i][j] - B->m[i][j];
        }
    }
    matrix->rows = A->rows;
    matrix->cols = A->cols;
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
                    YTS->m[i][0] = atof(record);
                } else {
                    XTS->m[i][j-1] = atof(record);
                }

                j++;
                record = strtok(NULL, ",");
            }
        }

        i++;
    }

    printMatrixMatlab(XTS);
}

void initializeMatrices() {
	int i, j;

	// Create input
    XTS = (Matrix *)malloc(sizeof(Matrix));
	XTS->m = (float **)malloc(N * sizeof(float *));
	for (i = 0; i < N; i++) {
		XTS->m[i] = (float *)malloc(M * sizeof(float));
	}
    XTS->rows = N;
    XTS->cols = M;

	// Create output
    YTS = (Matrix *)malloc(sizeof(Matrix));
	YTS->m = (float **)malloc(N * sizeof(float *));
    for (i = 0; i < N; i++) {
        YTS->m[i] = (float *)malloc(sizeof(float));
    }
    YTS->rows = N;
    YTS->cols = 1;

    // Retrieve data from csv
    readInXY(0, 10);

    // Create weight matrices
    WTS = (Matrix **)malloc(NUM_LAYERS * sizeof(Matrix **));
    for (i = 0; i < NUM_LAYERS; i++) {
        int numRows = (i == 0) ? M : LAYER_SIZES[i-1];
        float **w = (float **)malloc(numRows * sizeof(float *));
        for (j = 0; j < numRows; j++) {
            w[j] = (float *)calloc(LAYER_SIZES[i], sizeof(float));
        }

        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = numRows;
        matrix->cols = LAYER_SIZES[i];
        matrix->m = w;
        WTS[i] = matrix;

        matrixElementApply(WTS[i], increment);
    }

    // Create S matrices
    ZTS = (Matrix **)malloc((NUM_LAYERS - 1) * sizeof(Matrix **));
    for (i = 0; i < NUM_LAYERS - 1; i++) {
        float **z = (float **)malloc(N * sizeof(float *));
        for (j = 0; j < N; j++) {
            z[j] = (float *)calloc(LAYER_SIZES[i], sizeof(float));
        }

        Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
        matrix->rows = NUM_LAYERS - 1;
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

void feedForward(Matrix *out) {
    int layer;
    Matrix *in = XTS;
    Matrix *z;

    for (layer = 0; layer < NUM_LAYERS; layer++) {
        // Multiply Z with W to get S
        z = (Matrix *)malloc(sizeof(Matrix));
        matrixMatrixMultiply(in, WTS[layer], z);
        
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

void backPropagation(Matrix *estimation) {
    int layer, i;

    // Calculate the derivative of all S matrices
    for (layer = 0; layer < NUM_LAYERS - 1; layer++) {
        matrixElementApply(ZTS[layer], sigmoidDerivWhenAlreadyHaveSigmoid);
    }

    // Get the error
    Matrix *delta = (Matrix *)malloc(sizeof(Matrix));
    matrixMatrixElementSub(estimation, YTS, delta);

    printMatrix(delta);

    for (layer = NUM_LAYERS - 2; layer >= 0; layer--) {
        // Transpose W to multiply with delta
        Matrix *transW = (Matrix *)malloc(sizeof(Matrix));
        transpose(WTS[layer + 1], transW);

        Matrix *deltaW = (Matrix *)malloc(sizeof(Matrix));
        matrixMatrixMultiply(delta, transW, deltaW);

        // Element wise multiplication between deltaW and derivative of S
        matrixMatrixElementMultiply(deltaW, ZTS[layer], delta);

        // Calculate the weight updates
        if (layer >= 1) {
            // Calculate how much the weights need to be updated by

            // First transpose Z
            Matrix *transposedZ = (Matrix *)malloc(sizeof(Matrix));
            transpose(ZTS[layer - 1], transposedZ);

            // Now multiply Z with D
            Matrix *wUpdates = (Matrix *)malloc(sizeof(Matrix));
            matrixMatrixMultiply(transposedZ, delta, wUpdates);

            // Multiply by -eta
            matrixElementApply(wUpdates, multiplyByEta);
            printMatrix(wUpdates);

            // Calculated how much to update by.  Now apply to W to update
            matrixMatrixElementAdd(WTS[layer], wUpdates, WTS[layer]);

            // Free up temp matrices
            freeMatrix(transposedZ);
            freeMatrix(wUpdates);
        }

        // Free temp matrices
        freeMatrix(transW);
        freeMatrix(deltaW); 
    }

    // Free temporary matrices
    freeMatrix(delta);
}

int main(int argc, char** argv) {
    LAYER_SIZES = (int *)malloc(NUM_LAYERS * sizeof(int));
    LAYER_SIZES[0] = N;
    LAYER_SIZES[1] = 15;
    LAYER_SIZES[2] = 1;

	initializeMatrices();

    Matrix *out = (Matrix *)malloc(sizeof(Matrix));
    feedForward(out);
    backPropagation(out);

    freeMatrices();

    free(LAYER_SIZES);
}