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

/** 
 *  @brief function create new matrix and return pointer to it. 
 * 
 *  Function get size of output matrix <x, y>, check input data, allocate memory 
 *  (with calloc) for it and return pointer. If you use thread-safe calloc this
 *  function is thread-safe.
 * 
 *  On success, pointer to matrix is returned. If errors occurs, function return NULL;
 * 
 *  @param [in] x output matrix width.
 *  @param [in] y output matrix height
 */
matrix_t* matrix_create(uint32_t x, uint32_t y);

#endif

/**
*   @file  matrix.h
*   @brief Header file for matrix math.
*/