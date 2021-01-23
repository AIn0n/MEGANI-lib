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
mx_mp(const mx_t a, const mx_t b, mx_t* out, mx_mp_params params)
{
    uint32_t tra_y = a.x, tra_i = 1, limit = a.x;
    if(params & A)
    {
        tra_y = 1;
        tra_i = a.x;
        limit = a.y;
    }

    int32_t trb_x = 1, trb_i = b.x;
    if(params & B)
    {
        trb_x = b.x;
        trb_i = 1;
    }

    NN_TYPE val;
    for(uint32_t y = 0; y < out->y; ++y)
    {
        for(uint32_t x = 0; x < out->x; ++x)
        {
            val = 0;
            for(uint32_t i = 0; i < limit; ++i)
            {
                val += a.arr[i*tra_i + y*tra_y] * b.arr[x*trb_x + i*trb_i];
            }
            out->arr[x + y * out->x] = val; 
        }
    }
}