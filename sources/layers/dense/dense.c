#include "dense.h"

//STATIC FUNCTIONS

static void
dense_fill_rng(mx_t* values, const MX_TYPE min, const MX_TYPE max)
{
	const MX_TYPE diff = (max - min);
	for (MX_SIZE i = 0; i < values->size; ++i){
		MX_TYPE rand_val = (MX_TYPE) rand() / RAND_MAX;
		values->arr[i] = min + rand_val * diff;
	}
}

//PUBLIC FUNCTIONS

void
dense_forwarding(struct nn_layer_t* self, const mx_t * input)
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
	struct nn_layer_t*  self, 
	nn_array_t*         n, 
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

MX_SIZE
LAYER_DENSE(
	nn_array_t* nn,
	const MX_SIZE neurons,
	const act_func_t act_func,
	const MX_TYPE min,
	const MX_TYPE max)
{
	if (neurons < 1)
		return 1;
	const MX_SIZE in = (nn->size) ? nn->layers[nn->size - 1].out->x : nn->in_len;
	const MX_SIZE batch = nn->batch_len;
	struct nn_layer_t *layers = (struct nn_layer_t *) 
		realloc(nn->layers,sizeof(struct nn_layer_t) * (nn->size + 1));
	if (layers == NULL)
		return 2;
	nn->layers = layers;
	struct nn_layer_t* curr = &nn->layers[nn->size];
	nn->size++;
	curr->out = mx_create(neurons, batch);
	curr->delta = mx_create(neurons, batch);
	dense_data_t *data = (dense_data_t *) calloc(1, sizeof(dense_data_t));
	if (curr->out == NULL || curr->delta == NULL || data == NULL)
		return 2;
	data->act_func = act_func;
	data->val = mx_create(in, neurons);
	if (data->val == NULL)
		return 2;
	if (min && max)
		dense_fill_rng(data->val, min, max);
	if (nn->temp->size < in * neurons) {
		MX_TYPE* new_temp = (MX_TYPE *) 
			realloc(nn->temp->arr, in * neurons * sizeof(MX_TYPE));
		if (new_temp == NULL)
			return 2;
		nn->temp->arr = new_temp;
		mx_set_size(nn->temp, in, neurons);
	}
	curr->data		= (void *) data;
	curr->forwarding	= (& dense_forwarding);
	curr->backwarding	= (& dense_backwarding);
	curr->free_data		= (& dense_free_data);
	return 0;
}
/*
MX_SIZE
dense_setup(
	struct nn_layer_t* 	self, 
	const MX_SIZE		in, 
	const MX_SIZE		batch, 
	nn_params_t*		params, 
	const setup_params	purpose)
{
	if (purpose == DELETE) {
		dense_data_t* data = (dense_data_t *)self->data;
		if (data != NULL) {
			mx_destroy(data->val);
			free(data);
		}
		return 0;
	}
	self->out = mx_create(params->size, batch);
	self->delta = mx_create(params->size, batch);
	if (self->out == NULL || self->delta == NULL) 
		return 0;

	dense_data_t* data = (dense_data_t *)calloc(1, sizeof(dense_data_t));
	if (data == NULL) 
		return 0;

	data->act_func = params->activ_func;
	data->val = mx_create(in, params->size);
	if (data->val == NULL) 
		return 0;
		
	if (params->min && params->max) 
		dense_fill_rng(data->val, params);

	self->data = (void *) data;
	self->forwarding = (&dense_forwarding);
	self->backwarding= (&dense_backwarding);
	return in * params->size;
}*/