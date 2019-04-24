
#include <cstdio>

#include <cuda_runtime.h>
#include <math.h>

#include "matrix_factor_bias_gpu.cuh"


__device__ float dot(const float *Ui, const float *Vj, int k) {
    float result = 0;
    for (unsigned int i = 0; i < k; i++) {
        result += Ui[i] * Vj[i];
    }
    return result;
}

__device__ float getS(const float *Ui, const float *Vj, int k, int rating, float ai, float bj) {
    return (rating - MU - dot(Ui, Vj, k) - ai - bj);
}

__device__ float * add(float *arr1, float *arr2, float *output, int k) {
    for (unsigned int i = 0; i < k; i++) {
        output[i] = arr1[i] + arr2[i];
    }
}

__device__ float * sub(float *arr1, float *arr2, float *output, int k) {
    for (unsigned int i = 0; i < k; i++) {
        output[i] = arr1[i] - arr2[i];
    }
}

__device__ float * mult(float * arr1, float a, int k, float *output) {
    for (unsigned int i = 0; i < k; i++) {
        output[i] = arr1[i] * a;
    }
}

__device__ double * gradU(const float *Ui, const float *Vj, float s, float *output) {
    for (unsigned int i = 0; i < k; i++) {
        output[i] = REG * Ui[i] - Vj[i] * s;
    }
}

__device__ double * gradV(const float *Ui, const float *Vj, float s, float *output) {
    for (unsigned int i = 0; i < k; i++) {
        output[i] = REG * Vj[i] - Ui[i] * s;
    }
}

__device__ double gradA(float ai, float s) {
    return REG * ai - s;

}

__device__ double gradB(float bj, float s) {
    return REG * bj - s;
}





__global__
void sgd_kernel(const float **Ui, const float **Vi, const float *A, const float *B, float *rating, float *movie, float *user,
    float **newU, float **newV, float *newA, float *newB, int k, int num_pts) {


    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (thread_index < num_pts) {
        uint rating = rating[tid]
        uint movie = movie[tid]
        uint user = user[tid]

        float *Ui = Ui[user - 1];
        float *Vj = Vj[movie - 1];
        float a = A[user - 1];
        float b = B[movie - 1];

        float s = getS(Ui, Vj, k, rating, a, b);
        gradU(Ui, Vj, s, newU[user - 1]);
        gradV(Ui, Vj, s, newV[movie - 1]);
        gradA(a, s, newA[user - 1]);
        gradB(b, s, newB[movie - 1]);


        tid += blockDim.x * gridDim.x;
    }



void cudaCallSgdKernel(const float **Ui, const float **Vi, const float *A, const float *B, float *rating, float *movie, float *user,
    float **newU, float **newV, float *newA, float *newB, int k, int num_pts) {


        /* TODO 2: Call the max-finding kernel. */
    sgd_kernel<<<256, 256>>>(Ui, Vi, A, B, rating, movie, user, newU, newV, newA, newB, k, num_pts);
}
