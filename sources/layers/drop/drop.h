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

MX_SIZE drop_setup(struct nn_layer_t* layer, MX_SIZE in, MX_SIZE batch, nn_params_t* params, setup_params purpose);
void drop_forwarding(struct nn_layer_t* self, const mx_t * input);
void drop_backwarding(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta);

#endif