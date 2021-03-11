#include "mx.h"
#include <stdio.h>  //FOR DEBUG ONLY

mx_t* 
mx_create(uint32_t x, uint32_t y)
{
    if(!x || !y) return NULL;
    
    mx_t* output = (mx_t *)calloc(1, sizeof(mx_t));
    if(output == NULL) return NULL;

    output->size = x * y;
    output->arr = (NN_TYPE *)calloc(output->size, sizeof(NN_TYPE));
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
    MX_SIZE tra_y = a.x, tra_i = 1, limit = a.x;
    if(params & A)
    {
        tra_y = 1;
        tra_i = a.x;
        limit = a.y;
    }

    MX_SIZE trb_x = 1, trb_i = b.x;
    if(params & B)
    {
        trb_x = b.x;
        trb_i = 1;
    }

    NN_TYPE val;
    for(MX_SIZE y = 0; y < out->y; ++y)
    {
        for(MX_SIZE x = 0; x < out->x; ++x)
        {
            val = 0;
            for(MX_SIZE i = 0; i < limit; ++i)
            {
                val += a.arr[i*tra_i + y*tra_y] * b.arr[x*trb_x + i*trb_i];
            }
            out->arr[x + y * out->x] = val; 
        }
    }
}

void 
mx_hadamard(const mx_t a, const mx_t b, mx_t* out)
{
    for(MX_SIZE i = 0; i < out->size; ++i)
    {
        out->arr[i] = a.arr[i] * b.arr[i];
    }
}

void
mx_sub(const mx_t a, const mx_t b, mx_t* out)
{
    for(MX_SIZE i = 0; i < out->size; ++i)
    {
        out->arr[i] = a.arr[i] - b.arr[i];
    }
}

void 
mx_mp_num(mx_t* a, NN_TYPE num)
{
    for(MX_SIZE i = 0; i < a->size; ++i)
    {
        a->arr[i] *= num;
    }
}

void 
mx_hadam_lambda(mx_t* a, const mx_t b, NN_TYPE (*lambda)(NN_TYPE))
{
    for(MX_SIZE i = 0; i < a->size; ++i)
    {
        a->arr[i] *= (*lambda)(b.arr[i]);
    }
}

//---------------------------------DEBUG ONLY-----------------------------------

void
mx_print(const mx_t* a, char * name)
{
    printf("%s\n", name);
    if(a == NULL)
    {
        puts("NULL");
        return;
    }
    for(MX_SIZE i = 0; i < a->x * a->y; ++i)
    {
        if(i && !(i % a->x)) puts("");
        printf("%lf ", a->arr[i]);
    }
    puts("");
}