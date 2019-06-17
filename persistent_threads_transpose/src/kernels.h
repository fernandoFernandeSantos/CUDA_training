// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    16
#define BLOCK_ROWS  16

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR = TILE_DIM;

#define FLOOR(a,b) (a-(a%b))

// Compute the tile size necessary to illustrate performance cases for SM20+ hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X,512) * FLOOR(MATRIX_SIZE_Y, 512))
		/ (TILE_DIM * TILE_DIM);

#include "persistent_lib.h"

__device__ void process_data(float *odata, float *idata, int width,
		int height) {
//	// Handle to thread block group
//	cg::thread_block cta = cg::this_thread_block();
//	__shared__ float tile[TILE_DIM][TILE_DIM];
//
//	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
//	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
//
//	int index = xIndex + width * yIndex;
//
//	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
//		if (xIndex < width && yIndex < height) {
//			tile[threadIdx.y][threadIdx.x] = idata[index];
//		}
//	}
//
//	cg::sync(cta);
//
//	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
//		if (xIndex < height && yIndex < width) {
//			odata[index] = tile[threadIdx.y][threadIdx.x];
//		}
//	}

// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];

	int blockIdx_x, blockIdx_y;

	// do diagonal reordering
	if (width == height) {
		blockIdx_y = blockIdx.x;
		blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
	} else {
		int bid = blockIdx.x + gridDim.x * blockIdx.y;
		blockIdx_y = bid % gridDim.y;
		blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
	}

	// from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
	// and similarly for y

	int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
	int index_in = xIndex + (yIndex) * width;

	xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
	int index_out = xIndex + (yIndex) * height;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
		tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
	}

	cg::sync(cta);

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
		odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
	}
}

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copySharedMem(float *odata, float *idata, int width,
		int height) {
	PersistentKernel pk;
	while (pk.keep_working()) {
		pk.wait_for_work();
//		printf("PASSSOU %d\n", threadIdx.x + threadIdx.y);

		process_data(odata, idata, width, height);
//		printf("PASSSOU depois %d\n", threadIdx.x + threadIdx.y);

		pk.iteration_finished();
	}

}

