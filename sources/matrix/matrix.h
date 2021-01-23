#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h> //for uintN_t - better portability

/** @def    NN_TYPE
*   @brief  Type of variables which contain neural values.
*
*   By default it is "double".
*/
#define NN_TYPE double 

/** @struct mx_t 
*   @brief  Structure with matrix data and size.
*
*   I designed this struct as interpreted 1 dimensional array to make it as fast and small
*   as possible. To change the data type in the matrix, change NN_TYPE.
*   @see NN_TYPE
*/
typedef struct 
{
    NN_TYPE* arr;   /**< Array with matrix data. For type see NN_TYPE */
    uint32_t x;     /**< matrix width */
    uint32_t y;     /**< matrix height*/
}
mx_t;

/** @brief Matrix mutliply parameters.
 *  
 *  Set this to choose which matrix is transposed in mulitplication.
 */
typedef enum {  DEF = 0,    /**< default option */
                A = 1,      /**< first matrix is transposed */
                B = 2,      /**< second matrix is transposed */
                BOTH = 3    /**< both matrix are transposed */
                } mx_mp_params;

/** @brief Function create new matrix and return pointer to it. 
* 
*   Function get size of output matrix <x, y>, check input data, allocate memory 
*   (with calloc) for it and return pointer. Size of output matrix in memory depend on
*   NN_TYPE. If you use thread-safe calloc this function is thread-safe.
* 
*   On success, pointer to matrix is returned. If errors occurs, function return NULL;
* 
*   @param [in] x output matrix width.
*   @param [in] y output matrix height
*
*   @see NN_TYPE
 */
mx_t* mx_create(uint32_t x, uint32_t y);


/**
*   @brief Function free memory allocated for matrix.
*
*   Function get pointer to mx_t and frees both structure and NN_TYPE array.
*   Very possible that this is not thread-safe.
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
 *  mp_mx_params enum. Non thread-safe, input is not validated.
 * 
 *  @param [in] a first input matrix
 *  @param [in] b second output matrix
 *  @param [in] params transposing directing structure
 *  @param [out] out pointer to output matrix 
 *
 *  @see mx_mp_params
 */
void mx_mp(const mx_t a, const mx_t b, mx_t* out, mx_mp_params params);

#endif

/**
*   @file  matrix.h
*   @brief Header file for matrix math.
*/

/** @mainpage Caffeine-library documentation
 *  @section Introduction
 *  @subsection Goals
 * 
 *  At the end I want to have as small and fast as possible library for basic 
 *  nerual networks which can generate images. Library is almost depedency-free and 
 *  platform independent for easy porting to other architectures, systems etc.
 * 
 *  @section naming convention
 * 
 *  First word in every function is always a short name of thing which 
 *  we use (example mx -> matrix). Next one is operation, after that we can 
 *  have things like _nostdlib to mark that this func works without standard C lib.
 */