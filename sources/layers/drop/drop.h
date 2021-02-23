#ifndef _DROP_H_
#define _DROP_H_
#include "nn.h"

//TODO DOCS
typedef struct 
{
    mx_t*   mask;
    uint8_t drop_rate;
}
drop_data_t;

int32_t drop_setup(struct nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t* params, setup_params purpose);
void drop_forward(struct nn_layer_t* self, const mx_t * input);
void drop_backward(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta);

#endif