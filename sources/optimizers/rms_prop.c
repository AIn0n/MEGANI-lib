#include "MEGANI/optimizers/rms_prop.h"
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-07

void
rms_prop_update(
	void *opt_data,
	mx_t *weights, 
	mx_t *delta,
	const nn_size idx)
{
	rms_prop_data_t *data = (rms_prop_data_t *)opt_data;
	/* cache = rho * cache + (1 - rho) * delta^2 (elements-wise) */
	for (nn_size n = 0; n < data->caches[idx]->size; ++n) {
		data->caches[idx]->arr[n] =
			data->caches[idx]->arr[n] * data->rho +
			(1 - data->rho) * (delta->arr[n] * delta->arr[n]);
		/* weights += -alpha * delta / (sqrt_cell(cache) + epsilon) */
		weights->arr[n] += -data->alpha * delta->arr[n] /
				   (sqrt(data->caches[idx]->arr[n]) + EPSILON);
	}
}

void
rms_prop_destroy(optimizer_t self)
{
	rms_prop_data_t *data = (rms_prop_data_t *)self.params;
	for (nn_size i = 0; i < self.size; ++i)
		mx_destroy(data->caches[i]);
	free(data->caches);
	free(data);
}

optimizer_t
rms_prop_create(nn_t *nn, mx_type alpha, mx_type rho)
{
	rms_prop_data_t *data = calloc(1, sizeof(*data));
	if (data == NULL || nn == NULL ||
	    !(data->caches = calloc(nn->len, sizeof(*data->caches))))
		return (optimizer_t){ 0 };
	data->alpha = alpha;
	data->rho = rho;

	for (nn_size i = 0; i < nn->len; ++i) {
		const mx_size x = nn->layers[i].weights->x;
		const mx_size y = nn->layers[i].weights->y;
		if (!(data->caches[i] = mx_create(x, y))) {
			for (nn_size j = 0; j < i; ++j)
				mx_destroy(data->caches[j]);
			free(data->caches);
		}
	}
	return (optimizer_t){
		.params = (void *)data,
		.update = rms_prop_update,
		.size = nn->len };
}
