#include "matrix.h"
#include <stdio.h>
#include <string.h>
#include <math.h>




// contructor
Matrix *newMatrix(int m, int n)
{
    Matrix *A = (Matrix *)malloc(sizeof(Matrix));
    A->m = (float *)malloc(sizeof(float[m][n]));
    A->rows = m;
    A->cols = n;
    A->subMatrix = 0;
    return A;
}

Matrix *newMatrixSub(int m, int n)
{
    Matrix *A = (Matrix *)malloc(sizeof(Matrix));
    A->rows = m;
    A->cols = n;
    A->subMatrix = 1;
    return A;
}

// deconstructor
void freeMatrix(Matrix *matrix)
{
    if (matrix->subMatrix == 0) free(matrix->m);
    free(matrix);
}



Matrix *matrixMatrixElementSub(Matrix *A, Matrix *B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    Matrix *C = newMatrix(A->rows, A->cols);

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->m[IDXM(C, i, j)] = A->m[IDXM(A, i, j)] - B->m[IDXM(B, i, j)];
        }
    }

    return C;
}


Matrix *matrixMatrixElementAdd(Matrix *A, Matrix *B) {
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch %dx%d %dx%d- matrixMatrixElementSub\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    Matrix *C = newMatrix(A->rows, A->cols);


    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->m[IDXM(C, i, j)] = A->m[IDXM(A, i, j)] + B->m[IDXM(B, i, j)];
        }
    }

    return C;
}




Matrix *matrixMatrixMultiply(Matrix *A, Matrix *B)
{
    if (A->cols != B->rows) {
        printf("Dimension mismatch: %dx%d %dx%d - matrixMatrixMultiply\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }

    // Malloc the matrix C
    Matrix *C = newMatrix(A->rows, B->cols);

    // Fill in values for C
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->m[IDXM(A, i, k)] * B->m[IDXM(B, k, j)];
            }
            C->m[IDXM(C, i, j)] = sum;
        }
    }
    return C;
}


Matrix *matrixMatrixElementMultiply(Matrix *A, Matrix *B)
{
    if (A->rows != B->rows || A->cols != B->cols) {
        printf("Dimension mismatch: %dx%d %dx%d - matrixMatrixElementMultiply\n", A->rows, A->cols, B->rows, B->cols);
        exit(1);
    }


    Matrix* C = newMatrix(A->rows, A->cols);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->m[IDXM(C, i, j)] = (A->m[IDXM(A, i, j)])*(B->m[IDXM(B, i, j)]);
        }
    }

    return C;
}


float matrixReduceSumPow(Matrix *A, int exponent)
{
    float sum = 0.0;

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            sum += pow(A->m[IDXM(A, i, j)], exponent);
        }
    }

    return sum;
}



Matrix *matrixTranspose(Matrix *in)
{
  // Allocate the space for the new array
  Matrix *A = newMatrix(in->cols, in->rows);

  for (int i = 0; i < in->cols; i++) {
    for (int j = 0; j < in->rows; j++) {
      A->m[IDXM(A, i, j)] = in->m[IDXM(in, j, i)];
    }
  }

  return A;
}


void matrixElementApply(Matrix *A, float(*f)(float)) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->m[IDXM(A,i,j)] = f(A->m[IDXM(A,i,j)]);
        }
    }
}


void matrixElementApplyArg(Matrix *A, float(*f)(float, void*), void *arg) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->m[IDXM(A,i,j)] = f(A->m[IDXM(A,i,j)], arg);
        }
    }
}


float setToConst(float val, void *c)
{
    return *((float*)c);
}



float multByConst(float val, void *c)
{
    return val*(*((float*)c));
}


float setTo0(float val)
{
    return 0;
}



float setTo1(float val)
{
    return 1;
}


float setToRand(float val)
{
    return (float)rand()/(RAND_MAX);
}


float sigmoid(float val) {
    return (float)((double)1/(double)(1 + exp(-val)));
}


float sigmoidDerivWhenAlreadyHaveSigmoid(float val)
{
    return val * (1 - val);
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
