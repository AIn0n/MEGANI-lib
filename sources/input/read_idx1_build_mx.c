#include "read_idx1_build_mx.h"
#include "def_mx_iter.h"
#include "commons.h"

#include <stdio.h>
#include <stdlib.h>

uint8_t
read_and_verify_idx1_header(FILE *f, int32_t *size)
{
	int32_t magic;
	if (fread(&magic, 1, I32_LEN, f) != I32_LEN
	  ||fread(size, 1, I32_LEN, f) != I32_LEN)
		return 1;

	reverse_bytes_int32(&magic);
	reverse_bytes_int32(size);

	return (magic != 2049 || *size < 0);
}

struct mx_iterator_t 
read_idx1_build_mx(const char *filename, const mx_size batch, mx_t* (*build_mx)(mx_size, uint8_t*))
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL || batch < 1)
		return (struct mx_iterator_t){0};

	int32_t size;
	if (read_and_verify_idx1_header(f, &size))
		return (struct mx_iterator_t){0};
	
	def_mx_iter_data_t *data = calloc(1, sizeof(*data));
	if (data == NULL)
		goto err_close_file;
	uint8_t *buffer = calloc(batch, sizeof(*buffer));
	const mx_size batch_count = size / batch;

	if (buffer == NULL || !(data->list = calloc(batch_count, sizeof(*data->list))))
		goto err_clean_data;
	mx_size n;
	for (n = 0; n < batch_count; ++n) {
		if (fread(buffer, U8_LEN, batch, f) != batch)
			goto err_clean_list;
		data->list[n] = build_mx(batch, buffer);
		if (data->list[n] == NULL)
			goto err_clean_list;
	}
	free(buffer);
	fclose(f);

	data->size = batch_count;
	return (struct mx_iterator_t) {
		.data = data, .has_next = def_iter_has_next,
		.next = def_iter_next, .reset = def_iter_reset
	};
err_clean_list:
	for (mx_size m = n; m >= 0; --m)
		mx_destroy(data->list[m]);
err_clean_data:
	free(data);
	free(data->list);
	free(buffer);
err_close_file:
	fclose(f);
	return (struct mx_iterator_t){0};
}
