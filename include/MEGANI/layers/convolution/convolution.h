#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_
#include "image_layer_data.h"
#include "nn.h"

mx_t*
get_im2col_idxs(
	const img_size_t krnl_size,
	const img_size_t in_size,
	const mx_size stride,
	const mx_size batch);

#endif
