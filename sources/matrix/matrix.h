#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stdint.h> //for uintN_t

#define NN_TYPE float //type of variables with neural values

typedef struct 
{
    NN_TYPE* arr;
    uint32_t x;
    uint32_t y;
} 
matrix_t;

#endif