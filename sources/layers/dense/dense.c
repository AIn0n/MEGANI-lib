#include "dense.h"	/* include dense related datatypes and mx.h header */

void
dense_forwarding(struct nl_t *self, const mx_t *input)
{
	//output = input * values ^T
    	mx_mp(*input, *self->weights, self->out, B);  
	//layer output = activation function ( layer output )
	const dense_data_t *data = self->data;
	if (*data->act_func.func_mx != NULL) 
		(*data->act_func.func_mx)(self->out);
}

void 
dense_backwarding(const nn_t *nn, const nn_size	idx, const mx_t	*prev_out)
{
	const struct nl_t *self = (nn->layers + idx);
	const dense_data_t* data = (dense_data_t *) self->data;
	//delta = delta o activation function ( output )
	if (data->act_func.func_cell != NULL)	
		mx_hadam_lambda(
			nn->delta[self->cache_idx], *self->out, data->act_func.func_cell);
	//temporary matrix is shared between layers so we had to change the size
	mx_set_size(nn->temp, self->weights->x, self->weights->y);
	//value delta = delta^T * previous output
	mx_mp(*nn->delta[self->cache_idx], *prev_out, nn->temp, A);
	if (idx) {
		mx_set_size(nn->delta[!self->cache_idx], self->weights->x, nn->batch_len);
		mx_mp(*nn->delta[self->cache_idx], *self->weights, nn->delta[!self->cache_idx], DEF);
	}
	nn->optimizer.update(nn->optimizer.params, self->weights, nn->temp, idx);
}

void
dense_free(struct nl_t *self)
{
	dense_data_t *data = self->data;
	free(data);
	data = NULL;
	mx_destroy(self->weights);
	mx_destroy(self->out);
}

void
LAYER_DENSE(
	nn_t* nn,
	const mx_size neurons,
	const act_func_t act_func,
	const mx_type min,
	const mx_type max)
{
/* increase size of neural layers array by one, in case of failure return false*/
	if (nn->error || neurons < 1 || !try_append_layers(nn))
		goto dense_err_exit;
/* check if layer is first one in neural network and calculate input size for it */
	const mx_size in = (nn->len) ? nn->layers[nn->len - 1].out->x : nn->in_len;
/* neural network structure have two matrices for delta, we need to decide which
 * one this layer will use, so we check which matrix is used by previous layer
 * and get opposite one.
 */
	const uint8_t curr_cache = (nn->len) ? !(nn->layers[nn->len - 1].cache_idx) : 0;
	struct nl_t *curr = &nn->layers[nn->len++];

/* check if delta is big enough for this layer purpose, if not - realocate it
 * and check realocation success
 */
	if (mx_recreate_if_too_small(nn->delta[curr_cache], neurons, nn->batch_len))
		goto dense_err_exit;

/* same thing like in above delta realocation code, difference is a fact that here
 * is only one temporary matrix in neural network structure
 */
	if ((curr->weights = mx_create(in, neurons)) == NULL
	   || mx_recreate_if_too_small(nn->temp, in, neurons))
		goto dense_err_exit;

	dense_data_t *data = calloc(1, sizeof(*data));
	if ((curr->out = mx_create(neurons, nn->batch_len)) == NULL || data == NULL)
		goto dense_err_exit;

	data->act_func = act_func;

/* if min and max are other than zero we fill layer weights with values between (min, max) */
	if (min != NN_ZERO || max != NN_ZERO)
		mx_fill_rng(curr->weights, min, max);

	curr->data			= (void *) data;
	curr->forwarding	= (& dense_forwarding);
	curr->backwarding	= (& dense_backwarding);
	curr->free_data		= (& dense_free);
	curr->cache_idx		= curr_cache;
	return;
dense_err_exit:
	nn->error = 1;
}
