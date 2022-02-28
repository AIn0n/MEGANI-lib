#include "nn.h"
#include "dense.h"

void
nn_destroy(nn_t* nn)
{
	if (nn == NULL)
		return;
	for (NN_SIZE i = 0; i < nn->len; ++i) {
		mx_destroy(nn->layers[i].delta);
		mx_destroy(nn->layers[i].out);
		nn->layers[i].free_data(nn->layers[i].data);
	}
	free(nn->layers);
	mx_destroy(nn->temp);
	free(nn);
}

nn_t*
nn_create(
	const MX_SIZE in_len,
	const MX_SIZE batch_len,
	const MX_TYPE alpha)
{
	nn_t* result = (nn_t *) calloc(1, sizeof(nn_t));
	if (in_len < 1 || batch_len < 1 || result == NULL)
		return NULL;
	result->alpha		= alpha;
	result->in_len		= in_len;
	result->batch_len	= batch_len;
	result->len		= 0;

	result->layers = (struct nl_t *) calloc(0, sizeof(struct nl_t));
	if (result->layers == NULL) {
		free(result);
		return NULL;
	}

	result->temp = mx_create(1, 1);
	if (result->temp == NULL) {
		free(result);
		free(result->layers);
	}
	return result;
}

void
nn_predict(nn_t* nn, const mx_t* input)
{
	const mx_t* prev_out = input;
	for (NN_SIZE i = 0; i < nn->len; ++i) {
		nn->layers[i].forwarding((nn->layers + i), prev_out);
		prev_out = nn->layers[i].out;
	}
}

void
nn_fit(nn_t* nn, const mx_t *input, const mx_t* output)
{
	nn_predict(nn, input);
	const NN_SIZE end = nn->len - 1;
	//delta = output - expected output (last layer case)
	mx_sub(*nn->layers[end].out, *output, nn->layers[end].delta);

	for (NN_SIZE i = end; i > 0; --i) {
		nn->layers[i].backwarding(
			(nn->layers + i), 
			nn, 
			nn->layers[i - 1].out, 
			nn->layers[i - 1].delta);
	}
	//vdelta = delta^T * input
	nn->layers->backwarding(nn->layers, nn, input, NULL);
}

//ACTIVATION FUNCS
//TODO: split activation funcs to other files or even folder

void relu_mx(mx_t *a)
{
	for (MX_SIZE i = 0; i < a->size; ++i)
		a->arr[i] = MAX(a->arr[i], NN_ZERO);
}

MX_TYPE 
relu_deriv_cell(MX_TYPE a) 
{
	return (MX_TYPE)((a > NN_ZERO) ? 1 : 0);
}