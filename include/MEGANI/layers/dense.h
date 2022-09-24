#ifndef _DENSE_H_
#define _DENSE_H_
#include "MEGANI/nn/nn.h"
#include "MEGANI/mx/mx.h"

//TODO DOCS
typedef struct {
	act_func_t act_func;
} dense_data_t;

void LAYER_DENSE(
	nn_t *nn,
	const mx_size neurons,
	const act_func_t act_func,
	const mx_type min,
	const mx_type max);
#endif
