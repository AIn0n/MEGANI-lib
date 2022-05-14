#include "bgd.h"

void 
bgd_optimize(mx_t *vdelta, mx_t *weights, void *params)
{
	const bgd_data_t *data = (bgd_data_t *) params;
	mx_mp_num(vdelta, data->alpha);
	mx_sub(*weights, *vdelta, weights);
}
