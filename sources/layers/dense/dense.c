#include "dense.h"

//STATIC FUNCTIONS

static void
dense_fill_rng(mx_t *values, const mx_type min, const mx_type max)
{
	const mx_type diff = (max - min);
	for (mx_size i = 0; i < values->size; ++i) {
		mx_type rand_val = (mx_type) rand() / RAND_MAX;
		values->arr[i] = min + rand_val * diff;
	}
}

//PUBLIC FUNCTIONS

void
dense_forwarding(struct nl_t *self, const mx_t *input)
{
	//output = input * values ^T
    	mx_mp(*input, *self->weights, self->out, B);  
	//layer output = activation function ( layer output )
	const dense_data_t* data = self->data;
	if (*data->act_func.func_mx != NULL) 
		(*data->act_func.func_mx)(self->out);  
}

void 
dense_backwarding(
	struct nl_t	*self, 
	nn_t		*nn, 
	const nn_size	even,
	const mx_t	*prev_out)
{
	const dense_data_t* data = (dense_data_t *) self->data;

	//delta = delta o activation function ( output )
	if (data->act_func.func_cell != NULL)	
		mx_hadam_lambda(
			nn->delta[even], *self->out, data->act_func.func_cell);
	//temporary matrix is shared between layers so we had to change the size
	mx_set_size(nn->temp, self->weights->x, self->weights->y);
	//value delta = delta^T * previous output
	mx_mp(*nn->delta[even], *prev_out, nn->temp, A);
	mx_set_size(nn->delta[!even], self->weights->x, nn->batch_len);
	mx_mp(*nn->delta[even], *self->weights, nn->delta[!even], DEF);
	nn->optimizer.update(nn->temp, self->weights, nn->optimizer.params);
}

void
dense_free_data(void* data)
{
	if (data != NULL)
		free(data);
}

uint8_t
mx_recreate(mx_t *mx, const mx_size x, const mx_size y)
{
	mx_type *new_arr = (mx_type *) realloc(mx->arr, x * y * sizeof(mx_type));
	if (new_arr == NULL)
		return 1;
	mx->arr = new_arr;
	mx_set_size(mx, x, y);
	return 0;
}

uint8_t
try_append_layers(nn_t *nn)
{
	struct nl_t *l = (struct nl_t *) 
		realloc(nn->layers, sizeof(struct nl_t) * (nn->len + 1));
	nn->layers = (l == NULL) ? nn->layers : l;
	return (l != NULL);
}

void
LAYER_DENSE(
	nn_t* nn,
	const mx_size neurons,
	const act_func_t act_func,
	const mx_type min,
	const mx_type max)
{
	if (nn->error)
		goto dense_err_exit;
/* check if layer is first one in neural network and calculate input size for it */
	const mx_size in = (nn->len) ? nn->layers[nn->len - 1].out->x : nn->in_len;
/* increase size of neural layers array by one, in case of failure return false*/
	if (neurons < 1 || !try_append_layers(nn))
		goto dense_err_exit;
/* neural network structure have two matrices for delta, we need to decide which
 * one this layer will use, so we check if index of current layer is even
 */
	const nn_size even = nn->len % 2;
	struct nl_t* curr = &nn->layers[nn->len++];
	curr->out = mx_create(neurons, nn->batch_len);
/* check if delta is big enough for this layer purpose, if not - realocate it 
 * and check realocation success
 */
	if (neurons * nn->batch_len > nn->delta[even]->size && 
	    mx_recreate(nn->delta[even], neurons, nn->batch_len))
		goto dense_err_exit;

	curr->weights = mx_create(in, neurons);

/* same thing like in above delta realocation code, difference is a fact that here
 * is only one temporary matrix in neural network structure (layer index doesn't matter now)
 */
	if (curr->weights == NULL ||
	   (nn->temp->size < in * neurons && mx_recreate(nn->temp, in, neurons)))
		goto dense_err_exit;

	dense_data_t *data = (dense_data_t *) calloc(1, sizeof(dense_data_t));
	if (curr->out == NULL || data == NULL)
		goto dense_err_exit;

	data->act_func = act_func;

/* if min and max are other than zero we fill layer weights with values between (min, max) */
	if (min && max)
		dense_fill_rng(curr->weights, min, max);

	curr->data		= (void *) data;
	curr->forwarding	= (& dense_forwarding);
	curr->backwarding	= (& dense_backwarding);
	curr->free_data		= (& dense_free_data);
	return;
dense_err_exit:
	nn->error = 1;
}
