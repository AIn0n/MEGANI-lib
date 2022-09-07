#include "convolution.h"
#include "image_layer_data.h"

/*
 * images in memory layout
 *       ||   z
 * ------++--------
 *       || 00 01 02
 * x*y*b || 03 04 05
 *       || 06 07 08
 *       || 09 10 11
 */


#include <stdio.h>

mx_size
count_x_kernels(const img_size_t in_size, const img_size_t krnl_size, const mx_size stride)
{
	return (in_size.x - krnl_size.x) / stride + 1;
}

mx_size
count_y_kernels(const img_size_t in_size, const img_size_t krnl_size, const mx_size stride)
{
	return (in_size.y - krnl_size.y) / stride + 1;
}

/*
void
add_convolution_layer(
	nn_t *nn,
	img_size_t input_size,
	const img_size_t kernel_size,
	const mx_size stride,
	const act_func_t act_func,
	const mx_type min,
	const mx_type max)
{
	if (nn->error || !try_append_layers(nn))
		goto conv_err_exit;

	if (nn->len) {
		img_data_t *prev_data = (img_data_t *) nn->layers[nn->len - 1].data;
		input_size = prev_data->size;
	}

	const uint8_t curr_cache = (nn->len) ? !(nn->layers[nn->len - 1].cache_idx) : 0;
	struct nl_t *curr = &nn->layers[nn->len++];

	const mx_size output_x = input_size.z * kernel_size.x * kernel_size.y;
	const mx_size output_y = 
		count_x_kernels(input_size, kernel_size, stride)
		* count_y_kernels(input_size, kernel_size, stride)
		* nn->batch_len;

conv_err_exit:
	nn->error = 1;
}
*/

mx_t*
get_im2col_idxs(
	const img_size_t krnl_size,
	const img_size_t in_size,
	const mx_size stride,
	const mx_size batch)
{
	const mx_size out_x = (in_size.x - krnl_size.x) / stride + 1;
	const mx_size out_y = (in_size.y  - krnl_size.y) / stride + 1;


	mx_t *res = mx_create(
		in_size.z * krnl_size.x * krnl_size.y, 
		batch * out_x * out_y);
	
	if (res == NULL)
		return NULL;


	for (mx_size b = 0; b < batch; ++b) {
		for (mx_size z = 0; z < in_size.z; ++z) {
			for (mx_size oy = 0; oy < out_y; ++oy) {
				for (mx_size ox = 0; ox < out_x; ++ox) {
					for (mx_size ky = 0; ky < krnl_size.y; ++ky) {
						for (mx_size kx = 0; kx < krnl_size.x; ++kx) {
							res->arr[(z * krnl_size.x * krnl_size.y) + kx + (ky * krnl_size.x) + ((b * out_x * out_y) + (oy * out_x) + ox) * res->x] = 
							z + (((ox * stride) + kx) * in_size.z) + (((oy * stride) + ky) * in_size.x * in_size.z) + (in_size.x * in_size.y * in_size.z * b);
						}
					}
				}
			}
		}
	}

	return res;
}
