#include "read_idx3.h"
#include "def_mx_iter.h"

#include <stdint.h>
#include <stdio.h>


static void
reverse_bytes_int32(int32_t *n)
{
	if(n == NULL) return;
	unsigned char *start = (unsigned char *) n;
	unsigned char *end = (unsigned char *) n + 3;
	unsigned char tmp;
	for(int i = 0; i < 2; ++i) {
		tmp = *end;
		*end = *start;
		*start = tmp;
		start++; end--;
	}
}

#define I32_LEN 4
#define U8_LEN 1

static inline uint8_t
read_and_verify_header_idx3(FILE *f, int32_t *size, int32_t *height, int32_t *width)
{
	int32_t magic;
	if (fread(&magic,	1, I32_LEN, f) != I32_LEN
	    || fread(size,	1, I32_LEN, f) != I32_LEN
	    || fread(height,	1, I32_LEN, f) != I32_LEN
	    || fread(width,	1, I32_LEN, f) != I32_LEN)
		return 1;

	reverse_bytes_int32(&magic);
	reverse_bytes_int32(size);
	reverse_bytes_int32(height);
	reverse_bytes_int32(width);
	
	return (magic != 2051 || !*width || !*height || !*size);
}

struct mx_iterator_t
read_idx3(const char *filename, const mx_size batch_len, const uint8_t vertical)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL || batch_len == 0)
		return (struct mx_iterator_t){0};

	int32_t size, height, width;
	if (read_and_verify_header_idx3(f, &size, &height, &width))
		goto close_file;

	const mx_size batch_count 	= size / batch_len;
	const mx_size batch_cells 	= width * height * batch_len;
	const mx_size x = (vertical) ? batch_cells : (mx_size) (width * height);
	const mx_size y = (vertical) ? 1 : batch_len;

	def_mx_iter_data_t *data = malloc(sizeof(*data));
	if (data == NULL)
		goto close_file;
	*data = (def_mx_iter_data_t) {.curr = 0, .size = batch_count, .list = NULL};

	if (!(data->list = malloc(sizeof(*data->list) * batch_count)))
		goto free_data_err;

	mx_size n;
	for (n = 0; n < batch_count; ++n) {
		data->list[n] = mx_create(x, y);
		if (data->list[n] == NULL)
			goto free_all_err;
		for (mx_size i = 0; i < batch_cells; ++i) {
			uint8_t byte;
			if (fread(&byte, 1, U8_LEN, f) != U8_LEN)
				goto free_all_err;
			data->list[n]->arr[i] = (mx_type) byte / 255;
		}
	}
	fclose(f);
	return (struct mx_iterator_t) {
		.data = data, .next = def_iter_next, 
		.has_next = def_iter_has_next, .reset = def_iter_reset
	};
free_all_err:
	for (int j = n; j > -1; --j)
		mx_destroy(data->list[j]);
	free(data->list);
free_data_err:
	free(data);
close_file:
	fclose(f);
	return (struct mx_iterator_t){0};
}
