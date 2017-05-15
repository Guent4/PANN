#ifndef MATRIX_H
#define MATRIX_H


#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <stdbool.h>

// i is row j is col
#define IDXM(M, I, J) ((I)*((M)->cols) + (J))

typedef struct {
    int rows;
    int cols;
    float *m;
    bool subMatrix;
} Matrix;

#ifdef __cplusplus
extern "C" {
#endif


// Matrix constructor
Matrix *newMatrix(int m, int n);
Matrix *newMatrixSub(int m, int n);

// Matrix desctructor
void freeMatrix(Matrix *matrix);

// Matrix helpters
Matrix *matrixTranspose(Matrix *in);
Matrix *matrixMatrixMultiply(Matrix *A, Matrix *B);
Matrix *matrixMatrixElementSub(Matrix *A, Matrix *B);
Matrix *matrixMatrixElementAdd(Matrix *A, Matrix *B);

float matrixReduceSumPow(Matrix *A, int exponent);
void printMatrix(Matrix *matrix);
void printMatrixMatlab(Matrix *matrix);

Matrix *matrixMatrixElementMultiply(Matrix *A, Matrix *B);
void matrixElementApply(Matrix *A, float(*f)(float));

//matrix element apply methods
float setTo0(float val);
float setTo1(float val);
float setToRand(float val);
float sigmoid(float val);
float sigmoidDerivWhenAlreadyHaveSigmoid(float val);

void matrixElementApplyArg(Matrix *A, float(*f)(float, void*), void *arg);
float setToConst(float val, void *c);
float multByConst(float val, void *c);

#ifdef __cplusplus
}
#endif

#endif
