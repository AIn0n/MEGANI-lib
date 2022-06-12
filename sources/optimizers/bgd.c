#include "bgd.h"

void 
bgd_optimize(void *params, mx_t *vdelta, mx_t *weights, const nn_size idx)
{	
	(void)(idx); /* to stop compiler screaming about unused parameters ;) */
	const bgd_data_t *data = (bgd_data_t *) params;
	mx_mp_num(vdelta, data->alpha);
	mx_sub(*weights, *vdelta, weights);
}
