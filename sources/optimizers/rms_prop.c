#include "rms_prop.h"
#include <stdlib.h>

#define EPSILON 1e-7

void
rms_prop_update(void* opt_data, mx_t* weights, mx_t* delta, const nn_size idx)
{
	/* TODO: add functionality */
	rms_prop_data_t *cast_data = (rms_prop_data_t *) opt_data;
	/* cache = rho * cache + (1 - rho) * delta^2 (elements-wise) */
	mx_mp_num(cast_data->caches[idx], cast_data->rho);
	mx_elem_power_by_two(delta);
	mx_mp_num(delta, (1 - cast_data->rho));
	mx_add_to_first(cast_data->caches[idx], delta);
	(void) (weights);
	(void) (idx);
}

void
rms_prop_destroy(nn_size size, void* params)
{
	if (params == NULL)
		return;
	rms_prop_data_t *cast_data = (rms_prop_data_t *) params;
	for (nn_size i = 0; i < size; ++i)
		mx_destroy(cast_data->caches[i]);
	free(cast_data);
}

uint8_t
add_rms_prop(nn_t *nn, mx_type alpha, mx_type rho)
{
	rms_prop_data_t *data = calloc(1, sizeof(*data));
	if (data == NULL || nn == NULL ||
	   !(data->caches = calloc(nn->len, sizeof(*data->caches))))
		return 1;
	data->alpha = alpha;
	data->rho = rho;

	for (nn_size i = 0; i < nn->len; ++i) {
		const mx_size x = nn->layers[i].weights->x;
		const mx_size y = nn->layers[i].weights->y;
		if (!(data->caches[i] = mx_create(x, y))) {
			for (nn_size j = i; j >= 0; --j)
				mx_destroy(data->caches[j]);
			free(data->caches);
		}
	}
	nn->optimizer = (optimizer_t) {
		.params = (void *) data,
		.update = rms_prop_update,
		.params_destroy = rms_prop_destroy
	};
	return 0;
}
