#include "dense.h"

//STATIC FUNCTIONS

static void
dense_fill_rng(mx_t* values, const MX_TYPE min, const MX_TYPE max)
{
	const MX_TYPE diff = (max - min);
	for (MX_SIZE i = 0; i < values->size; ++i) {
		MX_TYPE rand_val = (MX_TYPE) rand() / RAND_MAX;
		values->arr[i] = min + rand_val * diff;
	}
}

//PUBLIC FUNCTIONS

void
dense_forwarding(struct nl_t* self, const mx_t * input)
{
	//output = input * values ^T
    	const dense_data_t* data = self->data;
    	mx_mp(*input, *data->val, self->out, B);  
	//layer output = activation function ( layer output )
	if (*data->act_func.func_mx != NULL) 
		(*data->act_func.func_mx)(self->out);  
}

void 
dense_backwarding(
	struct nl_t*  self, 
	nn_t*         n, 
	const mx_t*         prev_out, 
	mx_t*               prev_delta)
{
	const dense_data_t* data = (dense_data_t *) self->data;

	//delta = delta o activation function ( output )
	if (data->act_func.func_cell != NULL)
		mx_hadam_lambda(self->delta, *self->out, data->act_func.func_cell);
	//temporary matrix is shared between layers so we had to change the size
	mx_set_size(n->temp, data->val->x, data->val->y);

	if (prev_delta != NULL)  //prev delta = curr delta * curr values
		mx_mp(*self->delta, *data->val, prev_delta, DEF);

	mx_mp(*self->delta, *prev_out, n->temp, A);   //value delta = delta^T * previous output
	mx_mp_num(n->temp, n->alpha);                 //value delta = value delta * alpha
	mx_sub(*data->val, *n->temp, data->val);      //values = values - vdelta
}

void
dense_free_data(void* data)
{
	dense_data_t *d = (dense_data_t *) data;
	if (data != NULL) {
		mx_destroy(d->val);
		free(d);
	}
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
append_layers(nn_t *nn)
{
	struct nl_t *l = (struct nl_t *) 
		realloc(nn->layers, sizeof(struct nl_t) * (nn->size + 1));
	nn->layers = (l == NULL) ? nn->layers : l;
	return (l == NULL);
}

bool
LAYER_DENSE(
	nn_t* nn,
	const MX_SIZE neurons,
	const act_func_t act_func,
	const MX_TYPE min,
	const MX_TYPE max)
{
	if (neurons < 1)
		return false;
	
	const MX_SIZE in = (nn->size) ? nn->layers[nn->size - 1].out->x : nn->in_len;
	const MX_SIZE batch = nn->batch_len;

	if (append_layers(nn))
		return false;
	struct nl_t* curr = &nn->layers[nn->size++];
	curr->out = mx_create(neurons, batch);
	curr->delta = mx_create(neurons, batch);

	dense_data_t *data = (dense_data_t *) calloc(1, sizeof(dense_data_t));
	if (curr->out == NULL || curr->delta == NULL || data == NULL)
		return false;
	data->act_func = act_func;
	data->val = mx_create(in, neurons);
	if (data->val == NULL  ||
	   (nn->temp->size < in * neurons && mx_recreate(nn->temp, in, neurons)))
		return false;
	if (min && max)
		dense_fill_rng(data->val, min, max);
	curr->data		= (void *) data;
	curr->forwarding	= (& dense_forwarding);
	curr->backwarding	= (& dense_backwarding);
	curr->free_data		= (& dense_free_data);
	return true;
}