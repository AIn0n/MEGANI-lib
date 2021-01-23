#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h> //for uintN_t - better portability

/** @def    NN_TYPE
*   @brief  Type of variables which contain neural values.
*
*   By default it is "double".
*/
#define NN_TYPE double 

/** @struct matrix_t 
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
matrix_t;

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
matrix_t* matrix_create(uint32_t x, uint32_t y);


/**
*   @brief Function free memory allocated for matrix.
*
*   Function get pointer to matrix_t and frees both structure and NN_TYPE array.
*   Very possible that this is not thread-safe.
*
*   @param [in] matrix pointer to matrix allocated by matrix_create function.
*
*   @see matrix_create
*/
void matrix_free(matrix_t *matrix);


#endif

/**
*   @file  matrix.h
*   @brief Header file for matrix math.
*/