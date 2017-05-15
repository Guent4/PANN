#include "pfeed_forward.h"


extern cublasHandle_t handle;
extern int NUM_LAYERS;
extern Matrix **WTS;
extern Matrix **ZTS;



Matrix *feedForward(Matrix *in)
{
    cudaStream_t stream_hd;
    cudaStream_t stream_dh;

    // create our streams
    cudaStreamCreate(&stream_hd);
    cudaStreamCreate(&stream_dh);

    int wts_max = 0; //find max number of elements
    int max_cols = in->cols;
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int tmp = WTS[layer]->cols*WTS[layer]->rows;
        wts_max = ( tmp > wts_max) ? tmp : wts_max;

        max_cols = (WTS[layer]->cols > max_cols) ? WTS[layer]->cols : max_cols;
    }

    float *dev_wts;
    float *dev_in;
    float *dev_z;
    float *dev_z_trans;
    cudaMalloc((void**)&dev_wts, wts_max*sizeof(float));
    cudaMalloc((void**)&dev_in, in->rows*max_cols*sizeof(float));
    cudaMalloc((void**)&dev_z, in->rows*max_cols*sizeof(float));
    cudaMalloc((void**)&dev_z_trans, in->rows*max_cols*sizeof(float));

    const float alpha = 1;
    const float beta = 0;


    const int in_rows = in->rows;
    int in_cols = in->cols;

    int wts_cols = WTS[0]->cols;
    int wts_rows = WTS[0]->rows;

    // this will load in transposed
    cublasSetMatrixAsync(in->cols, in->rows, sizeof(float),
            in->m, in->cols, dev_in, in->cols, stream_hd);

    // set to our stream
    cublasSetStream(handle, stream_hd);

    for (int layer = 0; layer < NUM_LAYERS; layer++) {

        wts_cols = WTS[layer]->cols;
        wts_rows = WTS[layer]->rows;

        if (in_cols != wts_rows) {
            printf("Error! Dimension mismatch in feedforward\n");
            exit(1);
        }

        // Load WTS[layer]  transposed
        cublasSetMatrixAsync(wts_cols, wts_rows, sizeof(float),
                WTS[layer]->m, wts_cols, dev_wts, wts_cols, stream_hd);

        // multiply
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
            in_rows, wts_cols, in_cols, &alpha, dev_in, in_cols,
            dev_wts, wts_cols, &beta, dev_z, in_rows);

        // only apply sigmoid if not last layer
        if (layer == NUM_LAYERS - 1) // last output layer
            break;


        int blocks = UINT_DIV_CEIL((wts_cols*in_rows), BLOCK_SIZE);

        //printf("Launching kernel with dim y: %d, dim x: %d\n", UINT_DIV_CEIL(wts_cols, dimBlock.x), UINT_DIV_CEIL(in_rows , dimBlock.y));

        cuda_matirxElementSigmoid<<<blocks, BLOCK_SIZE, 0, stream_hd>>>(dev_z, in_rows, wts_cols);


        cudaStreamSynchronize(stream_dh); //Make sure the copy to host buffer is clear

        // stupid transpose to put it into col ordering
        // dev_z is dev_z[in_rows][wts_cols] (stored in row ordering on device)
        // now dev_z_trans is wts_cols x in_rows (stored in row ordering on device)
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, wts_cols, in_rows, &alpha,
                dev_z, in_rows, &beta, dev_z, wts_cols, dev_z_trans, wts_cols);

        // wait till done transposing so we can start async copy back to host
        cudaStreamSynchronize(stream_hd);

        // now dev_z is WTS[layer]->cols x in->rows (stored in row ordering on device)
        cudaMemcpyAsync(ZTS[layer]->m, dev_z_trans,
            in_rows*wts_cols*sizeof(float), cudaMemcpyDeviceToHost, stream_dh); //eventually make async

        //swap
        float *tmp = dev_in;
        dev_in = dev_z_trans;
        dev_z_trans = tmp;

        // update col dims
        in_cols = wts_cols;
    }

    Matrix *z = newMatrix(in_rows, WTS[NUM_LAYERS-1]->cols);

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, wts_cols, in_rows, &alpha, dev_z,
        in_rows, &beta, dev_z, wts_cols, dev_z_trans, wts_cols);

    cudaStreamSynchronize(stream_dh);
    cudaStreamSynchronize(stream_hd);

    // now dev_z is WTS[layer]->cols x in->rows (stored in row ordering on device)
    cudaMemcpy(z->m, dev_z_trans, in_rows*wts_cols*sizeof(float), cudaMemcpyDeviceToHost); //eventually make async


    cudaFree (dev_wts);
    cudaFree (dev_in);
    cudaFree (dev_z);
    cudaFree (dev_z_trans);

    cudaStreamDestroy(stream_hd);
    cudaStreamDestroy(stream_dh);

    // feed through last layer
    return z;
}



__global__ void cuda_matirxElementSigmoid(float* A, int rows, int cols)
{
    int tid = blockIdx.x *blockDim.x + threadIdx.x;


    if (tid < rows*cols)
    {
        // using __expf for best performance
        A[IDX2C(tid%rows, tid/rows, rows)] = 1.0/(1.0 + __expf(-1*A[IDX2C(tid%rows, tid/rows, rows)]));
    }

}
