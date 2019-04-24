#include "matrix_factor_bias.h"
#include <cuda_runtime.h>
#include "matrix_factor_bias_gpu.cuh"
#include <stdio.h>

using namespace std;

Model::Model(
        int M,
        int N,
        int K,
        double eta,
        double reg,
        string filename,
        double mu,
        double eps,
        double max_epochs


    ) : params( { M, N, K, eta, reg, Data(filename), mu, eps, max_epochs}){

}

/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

double Model::trainErr() {
    vector<vector<int> >::iterator ptr;

    int k = 0;
    double loss_err = 0.0;

    while (this->params.Y.hasNext()) {
        vector<int> p = this->params.Y.nextLine();
        int i = p[0];
        int j = p[1];
        int y = p[3];

        loss_err += pow((y - this->params.mu - dot(U.col(i - 1), V.col(j - 1))
                - a[i - 1] - b[j - 1]), 2);

        k++;
    }

    return loss_err / k;
}

Model::~Model() {}

void Model::train() {
    // this->U = Mat<double>(this->params.K, this->params.M, fill::randu);
    // this->V = Mat<double>(this->params.K, this->params.N, fill::randu);
    // this->a = Col<double>(this->params.M, fill::randu);
    // this->b = Col<double>(this->params.N, fill::randu);
    // this->del_U = Col<double>(this->params.K, fill::zeros);
    // this->del_V = Col<double>(this->params.K, fill::zeros);
	//
    // this->U -= 0.5;
    // this->V -= 0.5;
    // this->a -= 0.5;
    // this->b -= 0.5;



	float **U;
	float **V;
	float *A;
	float *B;
	float **newU;
	float **newV;
	float *newA;
	float *newB;

	// Allocate GPU memory

	gpuErrchk(cudaMalloc((void **) &U, sizeof(float) * M * k));
	gpuErrchk(cudaMalloc((void **) &V, sizeof(float) * N * k));
	gpuErrchk(cudaMalloc((void **) &A, sizeof(float) * M));
	gpuErrchk(cudaMalloc((void **) &B, sizeof(float) * N));

	gpuErrchk(cudaMalloc((void **) &newU, sizeof(float) * M * k));
	gpuErrchk(cudaMalloc((void **) &newV, sizeof(float) * N * k));
	gpuErrchk(cudaMalloc((void **) &newA, sizeof(float) * M));
	gpuErrchk(cudaMalloc((void **) &newB, sizeof(float) * N));

	// Copy input data, assuming U is stored in inputU, etc.


	gpuErrchk(cudaMemcpy(U, inputU, sizeof(float) * M * k, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(V, inputV, sizeof(float) * N * k, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(A, inputA, sizeof(float) * M, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(B, inputB, sizeof(float) * N, cudaMemcpyHostToDevice));











    for (int e = 0; e < this->params.max_epochs; e++) {
        cout << "Running Epoch " << e << endl;

		// Call GPU kernel, assuming rating, movie, user are stored in some array

		cudaCallSgdKernel(U, V, A,
			B, rating, movie, user,
		    newU, newV, newA, newB, k, num_pts);

		// Then deal with copying data back, depending on how many epochs want to
		// run on GPU











        this->params.Y.reset();
        cout << "Error " << trainErr() << endl;
        this->params.Y.reset();
    }
}
