#include <time.h>
#include <cublas.h>
#include <cblas.h>
#include <stdio.h>

int main(int argc, char** argv)

{

	int nbIter = 10000;

	int m;

	int n = 128;

	for (int j = 0; j < 10; ++j) {

		m = 16 << j;
		n = 16 << j;

// n = m;

		printf("-------------\nEvaluating %i iterations for a matrix %ix%i\n",
				nbIter, m, n);

		float time;

		float *mat, *x, *y;

		float *data = (float*) malloc(sizeof(float) * m * n);

		for (int i = 0; i < m * n; ++i)

			data[i] = ((float) i) / ((float) (m * n));

		unsigned int timer = 0;

// cuda test

//		StopWatchInterface *timer=NULL;
//		sdkCreateTimer(&timer);
//		sdkResetTimer(&timer);
//		sdkStartTimer(&timer);

		(cudaMalloc((void**) &mat, m * n * sizeof(float)));

		(cudaMalloc((void**) &x, n * sizeof(float)));

		(cudaMalloc((void**) &y, m * sizeof(float)));

		(cudaMemcpy(mat, data, m * n * sizeof(float), cudaMemcpyHostToDevice));

		(cudaMemcpy(x, data, n * sizeof(float), cudaMemcpyHostToDevice));

		(cudaMemcpy(y, data, m * sizeof(float), cudaMemcpyHostToDevice));

//		(cutStartTimer(timer));
		clock_t start = clock();
		for (int i = 0; i < nbIter; ++i) {

			cublasSgemv('t', n, m, 1, mat, n, x, 1, 1, y, 1);

		}
		printf("GPU Time: %f (ms)\n",
				(clock() - start) * 1000 / (float) CLOCKS_PER_SEC);
		(cudaThreadSynchronize());

//		sdkStopTimer(&timer);
//		time = sdkGetTimerValue(&timer);
//		cutDeleteTimer(&timer);

//		time = cutGetTimerValue(timer);

// output results

//		printf("CUDA Time: %f (ms)\n", time);

		(cudaFree(mat));

		(cudaFree(x));

		(cudaFree(y));
//
//		(cutDeleteTimer(timer));

// cpu test

		mat = (float*) malloc(m * n * sizeof(float));

		x = (float*) malloc(n * sizeof(float));

		y = (float*) malloc(m * sizeof(float));

		memcpy(mat, data, m * n * sizeof(float));

		memcpy(x, data, n * sizeof(float));

		memcpy(y, data, m * sizeof(float));

		start = clock();

		for (int i = 0; i < nbIter; ++i) {

			cblas_sgemv(CblasColMajor, CblasTrans, n, m, 1, mat, n, x, 1, 1, y,
					1);

		}

		printf("CPU Time: %f (ms)\n",
				(clock() - start) * 1000 / (float) CLOCKS_PER_SEC);

		free(mat);

		free(x);

		free(y);

		free(data);

	}

}
