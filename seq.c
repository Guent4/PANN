#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

#define LINES 10
#define FEATURES 11

float **X;
float ***W;
float *Y;

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
        // Only include interested lines
        if (i >= starting && i < ending) {
            record = strtok(line, ",");
            
            j = 0;
            while (record != NULL) {
                if (j == 0) {
                    Y[i] = atof(record);
                } else {
                    // printf("%f\t", atof(record));
                    X[i][j-1] = atof(record);
                }

                j++;
                record = strtok(NULL, ",");
            }
        }

        i++;
    }

    printVector(Y, LINES);
    printMatrix(X, LINES, FEATURES);
}

void initializeMatrices() {
	int i;

	// Create input
	X = (float **)malloc(LINES * sizeof(float *));
	for (i = 0; i < LINES; i++) {
		X[i] = (float *)malloc(FEATURES * sizeof(float));
	}

	// Create output
	Y = (float *)malloc(LINES * sizeof(float));

    // Retrieve data from csv
    readInXY(0, 10);
}

void freeMatrices() {
    int i;

    // Free X
    for (i = 0; i < LINES; i++) {
        free(X[i]);
    }
    free(X);

    // Free Y
    free(Y);
}

int main(int argc, char** argv) {
	initializeMatrices();

    freeMatrices();
}