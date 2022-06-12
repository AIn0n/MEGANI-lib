#ifndef _DENSE_H_
#define _DENSE_H_
#include "nn.h"
#include "mx.h"

//TODO DOCS
typedef struct {
	act_func_t  act_func;
}
dense_data_t;

void dense_forwarding(struct nl_t* self, const mx_t * input);

void 
dense_backwarding(
	const nn_t *nn, 
	const nn_size idx,
	const nn_size even,
	const mx_t* prev_out);

void LAYER_DENSE(nn_t* nn, const mx_size neurons, const act_func_t act_func, const mx_type min, const mx_type max);
#endif
