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

void dense_forwarding(struct nn_layer_t* self, const mx_t * input);
void dense_backwarding(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta);
MX_SIZE dense_setup(struct nn_layer_t* layer, MX_SIZE in, MX_SIZE batch, nn_params_t* params, setup_params purpose);

#endif