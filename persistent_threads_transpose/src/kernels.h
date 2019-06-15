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
		int height){
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	__shared__ float tile[TILE_DIM][TILE_DIM];

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

	int index = xIndex + width * yIndex;

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		if (xIndex < width && yIndex < height) {
			tile[threadIdx.y][threadIdx.x] = idata[index];
		}
	}

	cg::sync(cta);

	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		if (xIndex < height && yIndex < width) {
			odata[index] = tile[threadIdx.y][threadIdx.x];
		}
	}
}

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copySharedMem(float *odata, float *idata, int width,
		int height) {
	PersistentKernel pk;
	while(pk.stop_working()){
		pk.wait_for_work();
		process_data(odata, idata, width, height);
		pk.complete_work();
	}
}

