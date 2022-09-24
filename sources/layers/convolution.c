#include "MEGANI/layers/convolution.h"

/*
 * images in memory layout
 *       ||   z
 * ------++--------
 *       || 00 01 02
 * x*y*b || 03 04 05
 *       || 06 07 08
 *       || 09 10 11
 */

static inline mx_size
count_x_kernels(
	const img_size_t in_size,
	const img_size_t krnl_size,
	const mx_size stride)
{
	return (in_size.x - krnl_size.x) / stride + 1;
}

static inline mx_size
count_y_kernels(
	const img_size_t in_size,
	const img_size_t krnl_size,
	const mx_size stride)
{
	return (in_size.y - krnl_size.y) / stride + 1;
}

static inline mx_size
count_reshaped_input_x(const img_size_t in_size, const img_size_t krnl_size)
{
	return in_size.z * krnl_size.x * krnl_size.y;
}

static inline mx_size
count_reshaped_input_y(
	const mx_size x_krnl_count,
	const mx_size y_krnl_count,
	const mx_size batch)
{
	return x_krnl_count * y_krnl_count * batch;
}

/* TODO:do something to make this more readable */
static mx_size *
get_im2col_idxs(
	const img_size_t krnl_size,
	const img_size_t in_size,
	const mx_size stride,
	const mx_size batch)
{
	const mx_size out_x = count_x_kernels(in_size, krnl_size, stride),
		      out_y = count_y_kernels(in_size, krnl_size, stride);
	const size_t res_x = count_reshaped_input_x(in_size, krnl_size),
		     res_y = count_reshaped_input_y(out_x, out_y, batch);
	mx_size *res = calloc(res_x * res_y, sizeof(*res));
	if (res == NULL)
		return NULL;

	for (mx_size b = 0; b < batch; ++b) {
		for (mx_size z = 0; z < in_size.z; ++z) {
			for (mx_size oy = 0; oy < out_y; ++oy) {
				for (mx_size ox = 0; ox < out_x; ++ox) {
					for (mx_size ky = 0; ky < krnl_size.y; ++ky) {
						for (mx_size kx = 0; kx < krnl_size.x; ++kx) {
							res[(z * krnl_size.x * krnl_size.y) + kx + (ky * krnl_size.x) + ((b * out_x * out_y) + (oy * out_x) + ox) * res_x] = 
							z + (((ox * stride) + kx) * in_size.z) + (((oy * stride) + ky) * in_size.x * in_size.z) + (in_size.x * in_size.y * in_size.z * b);
						}
					}
				}
			}
		}
	}

	return res;
}

static void
fill_mx_by_idx_mx(const mx_t *src, mx_t *dst, const mx_size *idxs)
{
	for (mx_size n = 0; n < dst->size; ++n)
		dst->arr[n] = src->arr[idxs[n]];
}

void
conv_forwarding(struct nl_t *self, const mx_t *input)
{
	const img_data_t *data = (img_data_t *)self->data;
	const conv_data_t *conv_data = (conv_data_t *)data->data;

	fill_mx_by_idx_mx(input, conv_data->image_sections,
			  conv_data->im2col_idxs);

	mx_set_size(self->out, self->weights->y, conv_data->image_sections->y);
	mx_mp(*conv_data->image_sections, *self->weights, self->out, B);
	if (*conv_data->act_func.func_mx != NULL)
		(*conv_data->act_func.func_mx)(self->out);
}

void
conv_backwarding(const nn_t *nn, const nn_size idx, const mx_t *prev_out, optimizer_t opt)
{
	(void)(prev_out);
	const struct nl_t *self = (nn->layers + idx);
	const img_data_t *data = (img_data_t *)self->data;
	const conv_data_t *conv_data = (conv_data_t *)data->data;

	if (conv_data->act_func.func_cell != NULL)
		mx_hadam_lambda(nn->delta[self->cache_idx], *self->out,
				conv_data->act_func.func_cell);
	mx_set_size(nn->temp, self->weights->x, self->weights->y);
	mx_mp(*nn->delta[self->cache_idx], *conv_data->image_sections, nn->temp,
	      A);
	if (idx) {
		mx_set_size(nn->delta[!self->cache_idx], self->weights->x,
			    nn->batch_len);
		mx_mp(*nn->delta[self->cache_idx], *self->weights,
		      nn->delta[!self->cache_idx], DEF);
	}
	opt.update(opt.params, self->weights, nn->temp, idx);
}

void
conv_free(struct nl_t *self)
{
	img_data_t *d = (img_data_t *)self->data;
	conv_data_t *conv_data = (conv_data_t *)d->data;
	free(conv_data->im2col_idxs);
	mx_destroy(conv_data->image_sections);
	free(conv_data);
	free(d);
	mx_destroy(self->weights);
	mx_destroy(self->out);
}

/* TODO:this function should be splited into a few smaller, at this moment it is
 * 	totally unreadable mess
 */
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
		img_data_t *prev_data =
			(img_data_t *)nn->layers[nn->len - 1].data;
		input_size = prev_data->size;
	}

	const uint8_t curr_cache =
		(nn->len) ? !(nn->layers[nn->len - 1].cache_idx) : 0;
	struct nl_t *curr = &nn->layers[nn->len++];

	const mx_size x_krnl_count =
		count_x_kernels(input_size, kernel_size, stride);
	const mx_size y_krnl_count =
		count_y_kernels(input_size, kernel_size, stride);
	const mx_size output_x = kernel_size.z;
	const mx_size output_y = x_krnl_count * y_krnl_count * nn->batch_len;

	if (mx_recreate_if_too_small(nn->delta[curr_cache], output_x, output_y))
		goto conv_err_exit;

	img_data_t *data = calloc(1, sizeof(*data));
	conv_data_t *conv_data = calloc(1, sizeof(*conv_data));
	if ((curr->out = mx_create(output_x, output_y)) == NULL ||
	    data == NULL || conv_data == NULL)
		goto conv_err_exit;

	conv_data->act_func = act_func;

	if ((conv_data->im2col_idxs = get_im2col_idxs(
		     kernel_size, input_size, stride, nn->batch_len)) == NULL)
		goto conv_err_exit;

	if ((conv_data->image_sections = mx_create(
		     count_reshaped_input_x(input_size, kernel_size),
		     count_reshaped_input_y(x_krnl_count, y_krnl_count,
					    nn->batch_len))) == NULL)
		goto conv_err_exit;

	if ((curr->weights =
		     mx_create(kernel_size.x * kernel_size.y * input_size.z,
			       kernel_size.z)) == NULL ||
	    mx_recreate_if_too_small(nn->temp, curr->weights->x,
				     curr->weights->y))
		goto conv_err_exit;

	/* if min and max are other than zero we fill layer weights with random values between <min, max> */
	if (min != NN_ZERO || max != NN_ZERO)
		mx_fill_rng(curr->weights, min, max);

	data->size = (img_size_t){ 
		.x = x_krnl_count,
		.y = y_krnl_count,
		.z = kernel_size.z };
	data->data = (void *)conv_data;
	curr->data = (void *)data;
	curr->backwarding = &conv_backwarding;
	curr->forwarding = &conv_forwarding;
	curr->free_data = &conv_free;
	curr->cache_idx = curr_cache;
	return;
conv_err_exit:
	nn->error = 1;
}
