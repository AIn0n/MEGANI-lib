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
	struct nl_t*  self, 
	nn_t*         nn, 
	const NN_SIZE idx,
	const mx_t*	prev_out);
bool LAYER_DENSE(nn_t* nn, const MX_SIZE neurons, const act_func_t act_func, const MX_TYPE min, const MX_TYPE max);
#endif
