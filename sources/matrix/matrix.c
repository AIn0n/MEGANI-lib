#include <stdlib.h>
#include "matrix.h"

mx_t* 
mx_create(uint32_t x, uint32_t y)
{
    if(!x || !y) return NULL;
    
    mx_t* output = (mx_t *)calloc(1, sizeof(output));
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

void
mx_destroy(mx_t *mx)
{
    if(mx == NULL) return;
    if(mx->arr != NULL) free(mx->arr);
    free(mx);
}

void 
mx_mp(const mx_t a, const mx_t b, mx_t* out, uint8_t trnsp_a, uint8_t trnsp_b)
{
    
}