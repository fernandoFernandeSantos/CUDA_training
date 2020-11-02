#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <vector>
#include <iostream>

#ifdef RD_WG_SIZE_0_0                                                            
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)                                                      
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)                                                        
#define BLOCK_SIZE RD_WG_SIZE
#else                                                                                    
#define BLOCK_SIZE 16
#endif                                                                                   

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void fatal(char *s) {
	fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

	int i, j, index = 0;
	FILE *fp;
	char str[STR_SIZE];

	if ((fp = fopen(file, "w")) == 0)
		printf("The file was not opened\n");

	for (i = 0; i < grid_rows; i++)
		for (j = 0; j < grid_cols; j++) {

			sprintf(str, "%d\t%g\n", index, vect[i * grid_cols + j]);
			fputs(str, fp);
			index++;
		}

	fclose(fp);
}

template<typename double_t, typename single_t>
void compareOutputHost(std::vector<double_t> &vectDouble,
		std::vector<single_t> &vectSingle) {
	single_t max_relative = -99999;
	single_t min_relative = 99999;

	for (int i = 0; i < vectDouble.size(); i++) {
		auto dt = vectDouble[i];
		auto st = vectSingle[i];
		auto diff = (st - single_t(dt)) / st;
		max_relative = std::max(max_relative, diff);
		min_relative = std::min(min_relative, diff);
	}

	std::cout << "Max relative error on host " << max_relative << std::endl;
	std::cout << "Min relative error on host " << min_relative << std::endl;
}

template<typename double_t, typename single_t> __global__
void compareOutputGPU(double_t *vectDouble, single_t *vectSingle) {
	auto index = threadIdx.x;
	auto dt = vectDouble[index];
	auto st = vectSingle[index];

}

void readinput(double *vectDouble, float *vect, int grid_rows, int grid_cols,
		char *file) {

	int i, j;
	FILE *fp;
	char str[STR_SIZE];
	float val;

	if ((fp = fopen(file, "r")) == 0)
		printf("The file was not opened\n");

	for (i = 0; i <= grid_rows - 1; i++)
		for (j = 0; j <= grid_cols - 1; j++) {
			fgets(str, STR_SIZE, fp);
			if (feof(fp))
				fatal("not enough lines in file");
			//if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
			if ((sscanf(str, "%f", &val) != 1))
				fatal("invalid file format");
			vect[i * grid_cols + j] = val;
			vectDouble[i * grid_cols + j] = val;
		}

	fclose(fp);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

template<typename real_t>
__global__ void calculate_temp(int iteration,  //number of iteration
		real_t *power,   //power input
		real_t *temp_src,    //temperature input/output
		real_t *temp_dst,    //temperature input/output
		int grid_cols,  //Col of grid
		int grid_rows,  //Row of grid
		int border_cols,  // border offset
		int border_rows,  // border offset
		real_t Cap,      //Capacitance
		real_t Rx, real_t Ry, real_t Rz, real_t step, real_t time_elapsed) {

	__shared__ real_t temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ real_t temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

	real_t amb_temp = real_t(80.0f);
	real_t step_div_Cap;
	real_t Rx_1, Ry_1, Rz_1;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	step_div_Cap = step / Cap;

	Rx_1 = 1 / Rx;
	Ry_1 = 1 / Ry;
	Rz_1 = 1 / Rz;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
	int small_block_rows = BLOCK_SIZE - iteration * 2;        //EXPAND_RATE
	int small_block_cols = BLOCK_SIZE - iteration * 2;        //EXPAND_RATE

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkY = small_block_rows * by - border_rows;
	int blkX = small_block_cols * bx - border_cols;
	int blkYmax = blkY + BLOCK_SIZE - 1;
	int blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	int yidx = blkY + ty;
	int xidx = blkX + tx;

	// load data if it is within the valid input range
	int loadYidx = yidx, loadXidx = xidx;
	int index = grid_cols * loadYidx + loadXidx;

	if (IN_RANGE(loadYidx, 0,
			grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
		temp_on_cuda[ty][tx] = temp_src[index]; // Load the temperature data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index]; // Load the power data from global memory to shared memory
	}
	__syncthreads();

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows - 1) ?
	BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) :
												BLOCK_SIZE - 1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols - 1) ?
	BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) :
												BLOCK_SIZE - 1;

	int N = ty - 1;
	int S = ty + 1;
	int W = tx - 1;
	int E = tx + 1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (int i = 0; i < iteration; i++) {
		computed = false;
		if ( IN_RANGE(tx, i + 1, BLOCK_SIZE-i-2) &&
		IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&
		IN_RANGE(tx, validXmin, validXmax) &&
		IN_RANGE(ty, validYmin, validYmax)) {
			computed = true;
			temp_t[ty][tx] = temp_on_cuda[ty][tx]
					+ step_div_Cap
							* (power_on_cuda[ty][tx]
									+ (temp_on_cuda[S][tx] + temp_on_cuda[N][tx]
											- 2.0 * temp_on_cuda[ty][tx]) * Ry_1
									+ (temp_on_cuda[ty][E] + temp_on_cuda[ty][W]
											- 2.0 * temp_on_cuda[ty][tx]) * Rx_1
									+ (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed)	 //Assign the computation range
			temp_on_cuda[ty][tx] = temp_t[ty][tx];
		__syncthreads();
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		temp_dst[index] = temp_t[ty][tx];
	}
}

/*
 compute N time steps
 */
template<typename real_t>
int compute_tran_temp(real_t *MatrixPower, real_t *MatrixTemp[2], int col,
		int row, int total_iterations, int num_iterations, int blockCols,
		int blockRows, int borderCols, int borderRows) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);

	real_t grid_height = chip_height / row;
	real_t grid_width = chip_width / col;

	real_t Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	real_t Rx = grid_width / (real_t(2.0) * K_SI * t_chip * grid_height);
	real_t Ry = grid_height / (real_t(2.0) * K_SI * t_chip * grid_width);
	real_t Rz = t_chip / (K_SI * grid_height * grid_width);

	real_t max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	real_t step = PRECISION / max_slope;
	real_t t;
	real_t time_elapsed = real_t(0.001);

	int src = 1, dst = 0;

	for (t = 0; t < total_iterations; t += num_iterations) {
		int temp = src;
		src = dst;
		dst = temp;
		calculate_temp<<<dimGrid, dimBlock>>>(
				MIN(num_iterations, total_iterations - t), MatrixPower,
				MatrixTemp[src], MatrixTemp[dst], col, row, borderCols,
				borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
	}
	return dst;
}

void usage(int argc, char **argv) {
	fprintf(stderr,
			"Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n",
			argv[0]);
	fprintf(stderr,
			"\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
	fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr,
			"\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr,
			"\t<power_file> - name of the file containing the dissipated power values of each cell\n");
	fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char** argv) {
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

	run(argc, argv);

	return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
	int size;
	int grid_rows, grid_cols;
//	float *FilesavingTempFloat, *FilesavingPowerFloat, *MatrixOutFloat;
//	double *FilesavingTempDouble, *FilesavingPowerDouble, *MatrixOutDouble;

	std::vector<float> FilesavingTempFloat, FilesavingPowerFloat,
			MatrixOutFloat;
	std::vector<double> FilesavingTempDouble, FilesavingPowerDouble,
			MatrixOutDouble;

	char *tfile, *pfile, *ofile;

	int total_iterations = 60;
	int pyramid_height = 1; // number of iterations

	if (argc != 7)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[1])) <= 0
			|| (pyramid_height = atoi(argv[2])) <= 0 || (total_iterations =
					atoi(argv[3])) <= 0)
		usage(argc, argv);

	tfile = argv[4];
	pfile = argv[5];
	ofile = argv[6];

	size = grid_rows * grid_cols;

	/* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline	int borderCols = (pyramid_height) * EXPAND_RATE / 2;	int borderRows = (pyramid_height) * EXPAND_RATE / 2;	int smallBlockCol = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;	int smallBlockRow = BLOCK_SIZE - (pyramid_height) * EXPAND_RATE;
	int blockCols = grid_cols / smallBlockCol
			+ ((grid_cols % smallBlockCol == 0) ? 0 : 1);
	int blockRows = grid_rows / smallBlockRow
			+ ((grid_rows % smallBlockRow == 0) ? 0 : 1);

//	FilesavingTempFloat = (float *) malloc(size * sizeof(float));
//	FilesavingPowerFloat = (float *) malloc(size * sizeof(float));
//	MatrixOutFloat = (float *) calloc(size, sizeof(float));
//	if (!FilesavingPowerFloat || !FilesavingTempFloat || !MatrixOutFloat)
//		fatal("unable to allocate memory");

	FilesavingTempFloat.resize(size);
	FilesavingPowerFloat.resize(size);
	MatrixOutFloat.resize(size);

	FilesavingTempDouble.resize(size);
	FilesavingPowerDouble.resize(size);
	MatrixOutDouble.resize(size);

	printf(
			"pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
			pyramid_height, grid_cols, grid_rows, borderCols, borderRows,
			blockCols, blockRows, smallBlockCol, smallBlockRow);

	//READ FOR FLOAT and convert to DOUBLE
	readinput(FilesavingTempDouble.data(), FilesavingTempFloat.data(),
			grid_rows, grid_cols, tfile);
	readinput(FilesavingPowerDouble.data(), FilesavingPowerFloat.data(),
			grid_rows, grid_cols, pfile);

	float *MatrixTemp[2], *MatrixPower;
	cudaMalloc((void**) &MatrixTemp[0], sizeof(float) * size);
	cudaMalloc((void**) &MatrixTemp[1], sizeof(float) * size);
	cudaMemcpy(MatrixTemp[0], FilesavingTempFloat.data(), sizeof(float) * size,
			cudaMemcpyHostToDevice);

	cudaMalloc((void**) &MatrixPower, sizeof(float) * size);
	cudaMemcpy(MatrixPower, FilesavingPowerFloat.data(), sizeof(float) * size,
			cudaMemcpyHostToDevice);

	// -------------------------------------------------------------------------

	double *MatrixTempDouble[2], *MatrixPowerDouble;
	cudaMalloc((void**) &MatrixTempDouble[0], sizeof(double) * size);
	cudaMalloc((void**) &MatrixTempDouble[1], sizeof(double) * size);
	cudaMemcpy(MatrixTempDouble[0], FilesavingTempDouble.data(),
			sizeof(double) * size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &MatrixPowerDouble, sizeof(double) * size);
	cudaMemcpy(MatrixPowerDouble, FilesavingPowerDouble.data(),
			sizeof(double) * size, cudaMemcpyHostToDevice);
	// -------------------------------------------------------------------------

	printf("Start computing the transient temperature\n");
	int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows,
			total_iterations, pyramid_height, blockCols, blockRows, borderCols,
			borderRows);

	auto retDouble = compute_tran_temp(MatrixPowerDouble, MatrixTempDouble,
			grid_cols, grid_rows, total_iterations, pyramid_height, blockCols,
			blockRows, borderCols, borderRows);
	cudaDeviceSynchronize();

	printf("Ending simulation\n");
	cudaMemcpy(MatrixOutFloat.data(), MatrixTemp[ret], sizeof(float) * size,
			cudaMemcpyDeviceToHost);

	cudaMemcpy(MatrixOutDouble.data(), MatrixTempDouble[retDouble], sizeof(double) * size,
			cudaMemcpyDeviceToHost);

//	writeoutput(MatrixOutFloat, grid_rows, grid_cols, ofile);
	compareOutputHost(MatrixOutDouble, MatrixOutFloat);

	cudaFree(MatrixPower);
	cudaFree(MatrixTemp[0]);
	cudaFree(MatrixTemp[1]);

	cudaFree(MatrixPowerDouble);
	cudaFree(MatrixTempDouble[0]);
	cudaFree(MatrixTempDouble[1]);

//	free(MatrixOutFloat);
}
