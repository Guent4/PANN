#ifndef MATRIX_H

#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int rows;
    int cols;
    float *m;
} Matrix;


// Matrix constructor
Matrix *newMatrix(int m, int n);

// Matrix desctructor
void freeMatrix(Matrix *matrix);

// Matrix helpters
Matrix *matrixMatrixMultiply(Matrix *A, Matrix *B);
Matrix *matrixMatrixElementSub(Matrix *A, Matrix *B);
void printMatrix(Matrix *matrix);
void printMatrixMatlab(Matrix *matrix);


void matrixElementApply(Matrix *A, float(*f)(float));

//matrix element apply methods
float setTo1(float val);
float setToRand(float val);
float sigmoid(float val);


#define MATRIX_H
