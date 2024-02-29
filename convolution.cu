#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define FILTER_RADIUS 2
#define OUT_TILE_DIM 4
#define TILE_DIM OUT_TILE_DIM
#define IN_TILE_DIM (OUT_TILE_DIM + FILTER_RADIUS * 2)

__device__ __constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution(float *A, float *O, int width) {
	int out_col = blockIdx.x * blockDim.x + threadIdx.x;
	int out_row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0;
    int radius = FILTER_RADIUS;
	for (int f_row = 0; f_row < 2 * radius + 1; f_row++) {
		for (int f_col = 0; f_col < 2 * radius + 1; f_col++) {
			int in_row = out_row - radius + f_row;
			int in_col = out_col - radius + f_col;
            if (in_row >= 0 && in_row < width && in_col >= 0 && in_col < width) {
              sum += A[in_row * width + in_col] * F[f_row][f_col];
            }
		}
	}
	
	O[out_row * width + out_col] = sum;
	
}

__global__ void convolution_tiled_in_tile(float *A, float *O, int width) {
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float in_tile[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < width&& col >= 0 && col < width) {
        in_tile[threadIdx.y][threadIdx.x] = A[row * width + col];
    } else {
        in_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int tile_col = threadIdx.x - FILTER_RADIUS;
    int tile_row = threadIdx.y - FILTER_RADIUS;
    int radius = FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < width) {
        if (tile_col >= 0 && tile_col < OUT_TILE_DIM && tile_row >= 0
                && tile_row < OUT_TILE_DIM) {
            float sum = 0.0;
            for (int f_row = 0; f_row < 2 * radius + 1; f_row++) {
                for (int f_col = 0; f_col < 2 * radius + 1; f_col++) {
                  sum += F[f_row][f_col] * in_tile[tile_row + f_row][tile_col + f_col];
                }
            }
            O[row * width + col] = sum;
        }
    }
	
}

__global__ void convolution_tiled_using_cache(float *A, float *O, int width) {
	int col = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float in_tile[TILE_DIM][TILE_DIM];
    if (row < width && col < width) {
        in_tile[threadIdx.y][threadIdx.x] = A[row * width + col];
    } else {
        in_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    if (col < width && row < width) {
        float sum = 0.0;
        for (int f_row = 0; f_row < 2 * FILTER_RADIUS + 1; f_row++) {
            for (int f_col = 0; f_col < 2 * FILTER_RADIUS + 1; f_col++) {
                if (threadIdx.x-FILTER_RADIUS+f_col >= 0 &&
                    threadIdx.x-FILTER_RADIUS+f_col < TILE_DIM &&
                    threadIdx.y-FILTER_RADIUS+f_row >= 0 &&
                    threadIdx.y-FILTER_RADIUS+f_row < TILE_DIM) {
                  sum += F[f_row][f_col] * in_tile[threadIdx.y + f_row][threadIdx.x + f_col];
                } else {
                    if (row - FILTER_RADIUS + f_row >= 0 &&
                        row - FILTER_RADIUS + f_row < width &&
                        col - FILTER_RADIUS + f_col >= 0 &&
                        col - FILTER_RADIUS + f_col < width) {
                        int this_row = (row - FILTER_RADIUS + f_row) * width
                        int this_col = col - FILTER_RADIUS + f_col;
                        sum += F[f_row][f_col] * N[this_row + this_col];
                    }
                }
            }
        }

        O[row * width + col] = sum;
    }
}

int main() {
	int radius = FILTER_RADIUS;
	int size = 16;
	int f_size = radius * 2 + 1;

	float *A = (float *) malloc(sizeof(float) * size * size);
	float *F_h = (float *) malloc(sizeof(float) * f_size * f_size);
	float *O1 = (float *) malloc(sizeof(float) * size * size);
	float *O2 = (float *) malloc(sizeof(float) * size * size);
	float *O3 = (float *) malloc(sizeof(float) * size * size);

	for (int i = 0; i < size * size; i++) {
	    A[i] = i + 1;
	}

    for (int i = 0; i < f_size * f_size; i++) {
        F_h[i] = i;
    }

	float *d_A, *d_O1, *d_O2, *d_O3;

	cudaMalloc((void **)&d_A, sizeof(float) * size * size);
	cudaMalloc((void **)&d_O1, sizeof(float) * size * size);
	cudaMalloc((void **)&d_O2, sizeof(float) * size * size);
	cudaMalloc((void **)&d_O3, sizeof(float) * size * size);

	cudaMemcpy(d_A, A, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(F, F_h, f_size * f_size * sizeof(float));

    dim3 grid_dim(4, 4);
    dim3 block_dim(4, 4);
    dim3 block_dim2(IN_TILE_DIM, IN_TILE_DIM);
    dim3 block_dim3(TILE_DIM, TILE_DIM);

	convolution<<<grid_dim, block_dim>>>(d_A, d_O1, size);
	convolution_tiled_in_tile<<<grid_dim, block_dim2>>>(d_A, d_O2, size);
	convolution_tiled_using_cache<<<grid_dim, block_dim3>>>(d_A, d_O3, size);

	cudaMemcpy(O1, d_O1, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(O2, d_O2, sizeof(float) * size * size, cudaMemcpyDeviceToHost);
	cudaMemcpy(O3, d_O3, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    bool failed = false;
	for (int i = 0; i < size * size; i++) {
        if (abs(O1[i] - O2[i]) > 0.01) {
            failed = true;
        }

        if (abs(O2[i] - O3[i]) > 0.01) {
            failed = true;
        }
        printf("%f %f %f\n", O1[i], O2[i], O3[i]);
	}

    if (failed) {
        printf("failed\n");
    } else {
        printf("passed\n");
    }

	return 0;
}


