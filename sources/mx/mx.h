#ifndef _MX_H_
#define _MX_H_

#include <stdint.h> //for uintN_t - better portability
#include <stdlib.h>

/** @def    MX_TYPE
*   @brief  Type of variables which contain neural values.
*
*   By default it is "double".
*/
#define MX_TYPE double 

/** @def    NN_ZERO
 *  @brief  zero cast on MX_TYPE.
 */
#define NN_ZERO ((MX_TYPE)0)

/** @def MX_SIZE
 *  @brief Type of variables which contain matrix size, x, y. 
 * 
 *  By default it is uint32_t. I made this macro 'cause many times I used many 
 *  diffrent types for same tasks like iteration over matrix etc.
 */
#define MX_SIZE uint32_t

/** @struct mx_t 
*   @brief  Structure with matrix data and size.
*
*   I designed this struct as interpreted 1 dimensional array to make it as fast
*   and small as possible. To change the data type in matrices, change MX_TYPE.
*   @see MX_TYPE
*/
typedef struct 
{
    MX_TYPE* arr;   /**<    Array with matrix data */
    MX_SIZE x;      /**<    matrix width */
    MX_SIZE y;      /**<    matrix height*/
    MX_SIZE size;   /**<    matrix height multiplied by width. 
                            Nothing particular special but very often used */
}
mx_t;

/** @brief Matrix mutliply parameters.
 *  
 *  Set this to choose which matrix is transposed in mulitplication.
 */
typedef enum 
{  DEF = 0,    /**< default option */
    A = 1,      /**< first matrix is transposed */
    B = 2,      /**< second matrix is transposed */
    BOTH = 3    /**< both matrix are transposed */
} mx_mp_params;

/** @brief Function creates a new matrix and returns a pointer to it. 
* 
*   Function get size of the output matrix <x, y>, checks the input, allocates memory 
*   (with calloc) for it and returns the pointer. Size of output matrix in memory 
*   depend on MX_TYPE. If you use thread-safe calloc this function is thread-safe.
* 
*   On success, pointer to matrix is returned. If errors occurs, function return NULL;
* 
*   @param [in] x output matrix width.
*   @param [in] y output matrix height
*
*   @see MX_TYPE
 */
mx_t* mx_create(uint32_t x, uint32_t y);


/**
*   @brief Function free memory allocated for matrix.
*
*   Function get pointer to mx_t and frees both structure and MX_TYPE array.
*   This is not thread-safe.
*
*   @param [in] mx pointer to matrix allocated by matrix_create function.
*
*   @see mx_create
*/
void mx_destroy(mx_t *mx);

/**
 *  @brief Matrix multiply with transposing.
 * 
 *  Function get two matrices, multiply them and store output in out.
 *  You can choose which matrix is interpreted as transposed by configuring
 *  mx_mp_params enum. Not thread-safe, the input is not validated.
 * 
 *  @param [in] a first input matrix
 *  @param [in] b second output matrix
 *  @param [in] params transposing directing structure
 *  @param [out] out result of matrix multiplication (a * b = out) 
 *
 *  @see mx_mp_params
 */
void mx_mp(const mx_t a, const mx_t b, mx_t* out, mx_mp_params params);

/** @brief Matrix hadamard product function.
 * 
 *  Function takes two matrixes, executes hadamard protuct and stores
 *  output under <out> pointer. Not thread-safe and input is not validated.
 * 
 *  @param [in] a first input matrix
 *  @param [in] b second output matrix
 *  @param [out] out result of hadamard product (a o b = out)
 */
void mx_hadamard(const mx_t a, const mx_t b, mx_t* out);

/** @brief Matrix substraction function.
 * 
 *  Function takes two input matrices <a, b> and substract them. Result is
 *  store in <out> matrix pointer. Not thread-safe and the input is not validated.
 * 
 *  @param [in] a first input matrix
 *  @param [in] b second output matrix
 *  @param [out] out result of matrix substraction (a - b = out)
 */
void mx_sub(const mx_t a, const mx_t b, mx_t* out);

/** @brief Matrix multiplication by single number.
 * 
 *  Function assumes a pointer to <a> matrix and multiply every cell by <num>.
 *  Not thread-safe and input is not validated.
 * 
 *  @param [in] a input matrix
 *  @param [in] num every cell multiplier
 */
void mx_mp_num(mx_t* a, MX_TYPE num);

/** @brief matrix A = matrix A o lambda on cell( matrix B )
 * 
 * function uses lambda on every cell of second matrix and 
 * return to first matrix hadamard product.
 */
void mx_hadam_lambda(mx_t* a, const mx_t b, MX_TYPE (*lambda)(MX_TYPE));

void mx_print(const mx_t* a, char * name);

#endif

/**
*   @file  mx.h
*   @brief Header file for matrix math.
*/
