#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_
#include "image_layer_data.h"
#include "MEGANI/nn/nn.h"

typedef struct {
	act_func_t act_func;
	mx_size *im2col_idxs;
	mx_t *image_sections; /* this field can be cached but I don't have time to do that */
} conv_data_t;

void add_convolution_layer(
	nn_t *nn, img_size_t input_size,
	const img_size_t kernel_size, const mx_size stride,
	const act_func_t act_func, const mx_type min,
	const mx_type max);

#endif
