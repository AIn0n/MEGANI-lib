#include "nn.h"

void
nn_destroy(nn_t *nn)
{
	if (nn == NULL)
		return;
	if (nn->layers != NULL) {
		for (NN_SIZE i = 0; i < nn->len; ++i) {
			mx_destroy(nn->layers[i].out);
			mx_destroy(nn->layers[i].weights);
			nn->layers[i].free_data(nn->layers[i].data);
		}
		free(nn->layers);
	}
	mx_destroy(nn->temp);
	mx_destroy(nn->delta[0]);
	mx_destroy(nn->delta[1]);
	free(nn);
}

nn_t*
nn_create(const MX_SIZE in_len, const MX_SIZE batch_len, const MX_TYPE alpha) {
	nn_t *result = calloc(1, sizeof(*result));
	if (in_len < 1 || batch_len < 1 || result == NULL)
		return NULL;
	result->alpha		= alpha;
	result->in_len		= in_len;
	result->batch_len	= batch_len;

	result->temp = mx_create(1, 1);
	result->delta[0] = mx_create(1, 1);
	result->delta[1] = mx_create(1, 1);

	if (!result->temp || !result->delta[0] || !result->delta[1]) {
		nn_destroy(result);
		return NULL;
	}
	return result;
}

void
nn_predict(nn_t *nn, const mx_t *input)
{
	const mx_t *prev_out = input;
	for (NN_SIZE i = 0; i < nn->len; ++i) {
		nn->layers[i].forwarding((nn->layers + i), prev_out);
		prev_out = nn->layers[i].out;
	}
}

void
nn_fit(nn_t *nn, const mx_t *input, const mx_t *output)
{
	nn_predict(nn, input);
	const NN_SIZE end = nn->len - 1;
	NN_SIZE even = end % 2;
	/* delta = output - expected output (last layer case) */
	mx_sub(*nn->layers[end].out, *output, nn->delta[even]);
	for (NN_SIZE i = end; i > 0; --i, even = !even) {
		nn->layers[i].backwarding(
			(nn->layers + i), nn, even, nn->layers[i - 1].out);
	}
	/* vdelta = delta^T * input */
	nn->layers->backwarding(nn->layers, nn, even, input);
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
