#include "flatten.h"
#include "image_layer_data.h"

void
flatten_forwarding(struct nl_t *self, const mx_t *input)
{
	(void) (input); /* just to make compiler happy */
	const flatten_data_t *data = (flatten_data_t *) self->data;
	mx_set_size(self->out, data->post_x, data->post_y);
}

void
flatten_backwarding(const nn_t *nn, const nn_size idx, const mx_t *prev_out)
{
	(void) (prev_out); /* make compiler happy */
	const struct nl_t *self = nn->layers + idx;
	const flatten_data_t *data = (flatten_data_t *) self->data;
	mx_set_size(nn->delta[self->cache_idx], data->pre_x, data->pre_y);
}

void
flatten_free(struct nl_t *self)
{
	flatten_data_t *data = (flatten_data_t *) self->data;
	free(data);
}

void
add_flatten_layer(nn_t *nn)
{
	if (nn->error || !try_append_layers(nn))
		goto flat_err_exit;
	const uint8_t curr_cache = nn->layers[nn->len - 1].cache_idx;
	const struct nl_t *prev = &nn->layers[nn->len - 1];
	const img_data_t *prev_d = (img_data_t *) prev->data;
	const img_size_t prev_size = prev_d->size;
	struct nl_t *curr = &nn->layers[nn->len++];
	
	
	flatten_data_t *data = calloc(1, sizeof(*data));
	if (data == NULL)
		goto flat_err_exit;

	data->pre_x = prev->out->x;
	data->pre_y = prev->out->y;
	data->post_x = prev_size.x * prev_size.y * prev_size.z;
	data->post_y = nn->batch_len;

	curr->weights = prev->weights;
	curr->out = prev->out;
	curr->cache_idx = curr_cache;
	curr->data = data;
	mx_set_size(curr->out, data->post_x, data->post_y);

	curr->backwarding = flatten_backwarding;
	curr->forwarding = flatten_forwarding;
	curr->free_data = flatten_free;

	return;
flat_err_exit:
	nn->error = 1;
}