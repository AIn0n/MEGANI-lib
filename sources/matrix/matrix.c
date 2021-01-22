#include <stdlib.h>
#include "matrix.h"

matrix_t* 
matrix_create(uint32_t x, uint32_t y)
{
    if(!x || !y) return NULL;
    
    matrix_t* output = (matrix_t *)calloc(1, sizeof(output));
    if(output == NULL) return NULL;

    output->arr = (NN_TYPE *)calloc(x * y, sizeof(NN_TYPE));
    if(output->arr == NULL) 
    { 
        free(output); 
        return NULL;
    }

    output->x = x;
    output->y = y;
    return output;
}