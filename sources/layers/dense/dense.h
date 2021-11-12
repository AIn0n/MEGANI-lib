#ifndef _DENSE_H_
#define _DENSE_H_
#include "nn.h"
#include "mx.h"
//TODO DOCS
typedef struct {
	mx_t*       val;
	act_func_t  act_func;
}
dense_data_t;

void dense_forwarding(struct nn_layer_t* self, const mx_t * input);
void dense_backwarding(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta);
MX_SIZE dense_setup(struct nn_layer_t* layer, const MX_SIZE in, const MX_SIZE batch, nn_params_t* params, const setup_params purpose);

MX_SIZE
LAYER_DENSE(
	nn_array_t* nn,
	const MX_SIZE neurons,
	const act_func_t act_func,
	const MX_TYPE min,
	const MX_TYPE max);
#endif