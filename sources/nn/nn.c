#include "nn.h"

void
nn_destroy(nn_t *nn)
{
	if (nn == NULL)
		return;
	if (nn->layers != NULL) {
		for (nn_size i = 0; i < nn->len; ++i) {
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
nn_create(const mx_size in_len, const mx_size batch_len, optimizer_t optimizer)
{
	nn_t *result = calloc(1, sizeof(*result));
	if (in_len < 1 || batch_len < 1 || result == NULL)
		return NULL;
	result->in_len = in_len;
	result->batch_len = batch_len;
	result->optimizer = optimizer;

	if (!(result->temp = mx_create(1, 1)) 
	    || !(result->delta[0] = mx_create(1, 1)) 
	    || !(result->delta[1] = mx_create(1, 1))) {
		nn_destroy(result);
		return NULL;
	}
	return result;
}

void
nn_predict(nn_t *nn, const mx_t *input)
{
	const mx_t *prev_out = input;
	for (nn_size i = 0; i < nn->len; ++i) {
		nn->layers[i].forwarding((nn->layers + i), prev_out);
		prev_out = nn->layers[i].out;
	}
}

void
nn_fit(nn_t *nn, const mx_t *input, const mx_t *output)
{
	nn_predict(nn, input);
	const nn_size end = nn->len - 1;
	nn_size even = end % 2;
	/* delta = output - expected output (last layer case) */
	mx_sub(*nn->layers[end].out, *output, nn->delta[even]);
	for (nn_size i = end; i > 0; --i, even = !even) {
		nn->layers[i].backwarding(nn, i, even, nn->layers[i - 1].out);

	}
	/* vdelta = delta^T * input */
	nn->layers->backwarding(nn, 0, even, input);
}

//ACTIVATION FUNCS
//TODO: split activation funcs to other files or even folder

void relu_mx(mx_t *a)
{
	for (mx_size i = 0; i < a->size; ++i)
		a->arr[i] = MAX(a->arr[i], NN_ZERO);
}

mx_type 
relu_deriv_cell(mx_type a) 
{
	return (mx_type)((a > NN_ZERO) ? 1 : 0);
}
