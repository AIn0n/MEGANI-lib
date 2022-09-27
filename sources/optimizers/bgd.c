#include "MEGANI/optimizers/bgd.h"
#include "stdlib.h"

void
bgd_optimize(void *params, mx_t *weights, mx_t *vdelta, const nn_size idx)
{
	(void)(idx); /* to stop compiler screaming about unused parameters ;) */
	const bgd_data_t *data = (bgd_data_t *)params;
	mx_mp_num(vdelta, data->alpha);
	mx_sub(*weights, *vdelta, weights);
}

void
bgd_destroy(optimizer_t self)
{
	bgd_data_t *cast_data = (bgd_data_t *)self.params;
	free(cast_data);
	cast_data = NULL;
}

optimizer_t
bgd_create(nn_t *nn, const mx_type alpha)
{
	bgd_data_t *data = calloc(1, sizeof(*data));
	if (data == NULL || nn == NULL)
		return (optimizer_t){ 0 };
	data->alpha = alpha;
	return (optimizer_t){ 
		.update = bgd_optimize,
		.size = 1,
		.params = (void *)data };
}
