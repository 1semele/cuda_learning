#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void dot_product(float *A, float *B, float *C, int len) {
	float sum = 0.0;
	for (int i = 0; i < len; i++) {
		sum = A[i] * B[i];
	}
	C[0] = sum;
	
}


int main() {
	int len = 10;
	float *A = (float *) malloc(sizeof(float) * len);
	float *B = (float *) malloc(sizeof(float) * len);
	float *C = (float *) malloc(sizeof(float));

	for (int i = 0; i < len; i++) {
		A[i] = i;
		B[i] = i + 1;
	}

	float *d_A, *d_B, *d_C;

	cudaMalloc((void **)&d_A, sizeof(float) * len);
	cudaMalloc((void **)&d_B, sizeof(float) * len);
	cudaMalloc((void **)&d_C, sizeof(float) * len);

	cudaMemcpy(d_A, A, sizeof(float) * len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(float) * len, cudaMemcpyHostToDevice);

	dot_product<<<1, 1>>>(d_A, d_B, d_C, len);

	cudaMemcpy(C, d_C, sizeof(float) * len, cudaMemcpyDeviceToHost);

	printf("%f\n", C[0]);

	return 0;
}


