/**
*   @file  matrix.h
*   @brief Header file for matrix math.
*/
#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h> //for uintN_t - better portability

/** @def    NN_TYPE
*   @brief  Type of variables with neural values.
*/
#define NN_TYPE float 

/** @struct matrix_t 
*   @brief  Structure with matrix, data and size.
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

#endif