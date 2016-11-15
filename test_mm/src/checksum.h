/*
 * checksum.h
 *
 *  Created on: 15/11/2016
 *      Author: fernando
 */

#ifndef CHECKSUM_H_
#define CHECKSUM_H_


#define BLOCK_SIZE 32

#define N 6
#define ROWS_A N
#define COLLUMS_A N

#define ROWS_B N
#define COLLUMS_B N

#define VECTOR_SIZE_A COLLUMS_A * ROWS_A
#define VECTOR_SIZE_B COLLUMS_B * ROWS_B
#define VECTOR_SIZE_C ROWS_A * COLLUMS_B

#define MAX_THRESHOLD  0.0001
#define PRINT_TYPE long

#endif /* CHECKSUM_H_ */
