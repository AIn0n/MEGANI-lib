#include "dense.h"

//STATIC FUNCTIONS

static void
dense_fill_rng(mx_t *values, const MX_TYPE min, const MX_TYPE max)
{
	const MX_TYPE diff = (max - min);
	for (MX_SIZE i = 0; i < values->size; ++i) {
		MX_TYPE rand_val = (MX_TYPE) rand() / RAND_MAX;
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
	const NN_SIZE	even,
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

	mx_mp_num(nn->temp, nn->alpha);	//value delta = value delta * alpha
	mx_sub(*self->weights, *nn->temp, self->weights); //values = values - vdelta
}

void
dense_free_data(void* data)
{
	dense_data_t *d = (dense_data_t *) data;
	if (data != NULL)
		free(d);
}

bool
mx_recreate(mx_t *mx, const MX_SIZE x, const MX_SIZE y)
{
	MX_TYPE *new_arr = (MX_TYPE *) realloc(mx->arr, x * y * sizeof(MX_TYPE));
	if (new_arr == NULL)
		return true;
	mx->arr = new_arr;
	mx_set_size(mx, x, y);
	return false;
}

bool
try_append_layers(nn_t *nn)
{
	struct nl_t *l = (struct nl_t *) 
		realloc(nn->layers, sizeof(struct nl_t) * (nn->len + 1));
	nn->layers = (l == NULL) ? nn->layers : l;
	return (l != NULL);
}

bool
LAYER_DENSE(
	nn_t* nn,
	const MX_SIZE neurons,
	const act_func_t act_func,
	const MX_TYPE min,
	const MX_TYPE max)
{
	const MX_SIZE in = (nn->len) ? nn->layers[nn->len - 1].out->x : nn->in_len;
	const MX_SIZE batch = nn->batch_len;
	if (neurons < 1 || !try_append_layers(nn))
		return false;

	const NN_SIZE even = nn->len % 2;
	struct nl_t* curr = &nn->layers[nn->len++];
	curr->out = mx_create(neurons, batch);
	if (neurons * batch > nn->delta[even]->size && 
	    mx_recreate(nn->delta[even], neurons, batch))
		return false;

	curr->weights = mx_create(in, neurons);
	if (curr->weights == NULL ||
	   (nn->temp->size < in * neurons && mx_recreate(nn->temp, in, neurons)))
	   	return false;

	dense_data_t *data = (dense_data_t *) calloc(1, sizeof(dense_data_t));
	if (curr->out == NULL || data == NULL)
		return false;

	data->act_func = act_func;

	if (min && max)
		dense_fill_rng(curr->weights, min, max);
	curr->data		= (void *) data;
	curr->forwarding	= (& dense_forwarding);
	curr->backwarding	= (& dense_backwarding);
	curr->free_data		= (& dense_free_data);
	return true;
}
