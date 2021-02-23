#ifndef _DENSE_H_
#define _DENSE_H_
#include "nn.h"

//TODO DOCS
typedef struct 
{
    mx_t*       val;
    act_func_t  act_func;
} 
dense_data_t;

void dense_forward(struct nn_layer_t* self, const mx_t * input);
void dense_backward(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta);
int32_t dense_setup(struct nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t* params, setup_params purpose);

#endif