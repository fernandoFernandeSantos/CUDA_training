/*
 * sgemm_nn_64_16_16_16_4.h
 *
 *  Created on: 30/06/2019
 *      Author: fernando
 */

#ifndef SGEMM_NN_64_16_16_16_4_H_
#define SGEMM_NN_64_16_16_16_4_H_

void sgemm_N_N_64_16_16_16_4_special(cudaStream_t stream, float *C,
		const float *A, const float *B, int32_t m, int32_t n, int32_t k,
		int32_t lda, int32_t ldb, int32_t ldc, float alpha, float beta);

#endif /* SGEMM_NN_64_16_16_16_4_H_ */
