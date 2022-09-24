#ifndef _FLATTEN_H_
#define _FLATTEN_H_
#include "MEGANI/nn/nn.h"

typedef struct {
	mx_size pre_x, pre_y, post_x, post_y;
} flatten_data_t;

void add_flatten_layer(nn_t *nn);

#endif