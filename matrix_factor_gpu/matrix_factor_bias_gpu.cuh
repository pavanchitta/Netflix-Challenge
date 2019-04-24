
void cudaCallSgdKernel(const float **Ui, const float **Vi, const float *A,
	const float *B, float *rating, float *movie, float *user,
    float **newU, float **newV, float *newA, float *newB, int k, int num_pts);
