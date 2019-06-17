////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

// ----------------------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve performance
// through coalescing, removing shared memory bank conflicts, and eliminating partition
// camping.  Several of the kernels perform a copy, used to represent the best case
// performance that a transpose can achieve.
//
// Please see the whitepaper in the docs folder of the transpose project for a detailed
// description of this performance study.
// ----------------------------------------------------------------------------------------

#include <cooperative_groups.h>
#include <unistd.h>

namespace cg = cooperative_groups;
// Utilities and system includes
#include <helper_string.h>    // helper for string parsing
#include <helper_image.h>     // helper for image and data comparison
#include <helper_cuda.h>      // helper for cuda error checking functions

#include "kernels.h"

const char *sSDKsample = "Transpose";

// Number of repetitions used for timing.  Two sets of repetitions are performed:
// 1) over kernel launches and 2) inside the kernel over just the loads and stores

#define NUM_REPS  10

// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(std::vector<float>& gold, std::vector<float>& idata,
		const int size_x, const int size_y) {
	for (int y = 0; y < size_y; ++y) {
		for (int x = 0; x < size_x; ++x) {
			gold[(x * size_y) + y] = idata[(y * size_x) + x];
		}
	}
}

bool compare_data(std::vector<float>& gold, std::vector<float>& found,
		float threshold) {
	for (auto i = 0; i < gold.size(); i++) {
		float diff = fabs(gold[i] - found[i]);
		if (diff > threshold) {
			return false;
		}
	}
	return true;
}

void print(std::vector<float>& ptr, int x, int y) {
	for (auto i = 0; i < x; i++) {
		for (auto j = 0; j < y; j++) {
			std::cout << ptr[i * x + j] << " ";
		}
		std::cout << std::endl;
	}
}

void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &size_x,
		int &size_y, int max_tile_dim) {
	// set matrix size (if (x,y) dim of matrix is not square, then this will have to be modified
	if (checkCmdLineFlag(argc, (const char **) argv, "dimX")) {
		size_x = getCmdLineArgumentInt(argc, (const char **) argv, "dimX");

		if (size_x > max_tile_dim) {
			printf(
					"> MatrixSize X = %d is greater than the recommended size = %d\n",
					size_x, max_tile_dim);
		} else {
			printf("> MatrixSize X = %d\n", size_x);
		}
	} else {
		size_x = max_tile_dim;
		size_x = FLOOR(size_x, 512);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "dimY")) {
		size_y = getCmdLineArgumentInt(argc, (const char **) argv, "dimY");

		if (size_y > max_tile_dim) {
			printf(
					"> MatrixSize Y = %d is greater than the recommended size = %d\n",
					size_y, max_tile_dim);
		} else {
			printf("> MatrixSize Y = %d\n", size_y);
		}
	} else {
		size_y = max_tile_dim;
		size_y = FLOOR(size_y, 512);
	}
}

void showHelp() {
	printf("\n%s : Command line options\n", sSDKsample);
	printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
	printf(
			"> The default matrix size can be overridden with these parameters\n");
	printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
	printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

// ----
// main
// ----

int main(int argc, char **argv) {
	// Start logs
	printf("%s Starting...\n\n", sSDKsample);

	if (checkCmdLineFlag(argc, (const char **) argv, "help")) {
		showHelp();
		return 0;
	}

	int devID = findCudaDevice(argc, (const char **) argv);
	cudaDeviceProp deviceProp;

	// get number of SMs on this GPU
	checkCudaErrors(cudaGetDevice(&devID));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

	// compute the scaling factor (for GPUs with fewer MPs)
	float scale_factor, total_tiles;
	scale_factor = max(
			(192.0f
					/ (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
							* (float) deviceProp.multiProcessorCount)), 1.0f);

	printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
	printf("> SM Capability %d.%d detected:\n", deviceProp.major,
			deviceProp.minor);

	// Calculate number of tiles we will run for the Matrix Transpose performance tests
	int size_x, size_y, max_matrix_dim, matrix_size_test;

	matrix_size_test = 512;  // we round down max_matrix_dim for this perf test
	total_tiles = (float) MAX_TILES / scale_factor;

	max_matrix_dim = FLOOR((int)(floor(sqrt(total_tiles))* TILE_DIM),
			matrix_size_test);

	// This is the minimum size allowed
	if (max_matrix_dim == 0) {
		max_matrix_dim = matrix_size_test;
	}

	printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
			deviceProp.name, deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor)
					* deviceProp.multiProcessorCount);

	printf("> Compute performance scaling factor = %4.2f\n", scale_factor);

	// Extract parameters if there are any, command line -dimx and -dimy can override
	// any of these settings
	getParams(argc, argv, deviceProp, size_x, size_y, max_matrix_dim);

	if (size_x != size_y) {
		printf(
				"\n[%s] does not support non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n",
				sSDKsample, size_x, size_y);
		exit (EXIT_FAILURE);
	}

	if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
		printf(
				"[%s] Matrix size must be integral multiple of tile size\nExiting...\n\n",
				sSDKsample);
		exit (EXIT_FAILURE);
	}

	// execution configuration parameters
	dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM), threads(TILE_DIM,
	BLOCK_ROWS);

	if (grid.x < 1 || grid.y < 1) {
		printf("[%s] grid size computation incorrect in test \nExiting...\n\n",
				sSDKsample);
		exit (EXIT_FAILURE);
	}

	// size of memory required to store the matrix
	size_t element_size = size_x * size_y;
	size_t mem_size = static_cast<size_t>(sizeof(float) * size_x * size_y);

	if (2 * mem_size > deviceProp.totalGlobalMem) {
		printf(
				"Input matrix size is larger than the available device memory!\n");
		printf("Please choose a smaller size matrix\n");
		exit (EXIT_FAILURE);
	}

	// allocate host memory
//	float *h_idata = (float *) malloc(mem_size);
//	float *h_odata = (float *) malloc(mem_size);
//	float *transposeGold = (float *) malloc(mem_size);
	std::vector<float> h_idata(element_size);
	std::vector<float> h_odata(element_size);
	std::vector<float> gold(element_size);

	// allocate device memory
	float *d_idata, *d_odata;
	checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
	checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

	HostPersistentControler main_control(grid);

	// initialize host data
	for (int i = 0; i < (size_x * size_y); ++i) {
		h_idata[i] = i;
	}

	// copy host data to device
	checkCudaErrors(
			cudaMemcpy(d_idata, h_idata.data(), mem_size,
					cudaMemcpyHostToDevice));

	// Compute reference transpose solution
	computeTransposeGold(gold, h_idata, size_x, size_y);
	std::cout << std::boolalpha;
//	std::cout << "INPUT DATA " << std::endl;
//	print(h_idata, size_x, size_y);

	// print out common data for all kernels
	printf(
			"\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
			size_x, size_y, size_x / TILE_DIM, size_y / TILE_DIM, TILE_DIM,
			TILE_DIM, TILE_DIM, BLOCK_ROWS);

	// Clear error status
	checkCudaErrors(cudaGetLastError());

	std::cout << "New stream" << std::endl;

	std::cout << "Starting running kernel\n";
	checkCudaErrors(cudaPeekAtLastError());

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "GRID " << grid.x << " " << grid.y << " threads " << threads.x
			<< " " << threads.y << std::endl;

	copySharedMem<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);
	// Ensure no launch failure
	checkCudaErrors(cudaGetLastError());

	std::cout << "Trying persistent threads shared memory transpose"
			<< std::endl;
	size_t num_rep = 0;

	while (true) {
		std::cout << "Start processing" << std::endl;

		main_control.start_processing();

		main_control.wait_gpu();

		std::cout << "Copy memory back to the host" << std::endl;

		checkCudaErrors(
				cudaMemcpyAsync(h_odata.data(), d_odata, mem_size,
						cudaMemcpyDeviceToHost, main_control.st));

		main_control.sync_stream();
		bool res = compare_data(gold, h_odata, 0.0f);
		num_rep++;

		if (!res) {
			std::cout << "Process finished failed, iteration " << num_rep
					<< std::endl;
			std::cout << "GOLD" << std::endl;
			print(gold, size_x, size_y);
			std::cout << "COMPUTED" << std::endl;
			print(h_odata, size_x, size_y);

		} else {
			std::cout << "Process finished OK, iteration " << num_rep
					<< std::endl;

		}

		if (num_rep > NUM_REPS)
			break;
	}

	main_control.end_kernel();
	std::cout << "Releasing memory" << std::endl;

	cudaFree(d_idata);
	cudaFree(d_odata);

	std::cout << "Synchronizing the device" << std::endl;

	checkCudaErrors(cudaDeviceReset());

	printf("Test passed\n");
	exit (EXIT_SUCCESS);

}
