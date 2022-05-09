#include "read_idx3.h"

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

/* In case of some errors in reading function I just return empty iterator */
const mx_iterator empty_iterator = {.list = NULL, .size = 0, .curr = 0};

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

//TODO: make arguments constants, add newline character
mx_iterator
read_idx3(const char *filename, mx_size batch_len, uint8_t vertical)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL || batch_len == 0)
		return empty_iterator;

	int32_t size, height, width;
	if (read_and_verify_header_idx3(f, &size, &height, &width))
		goto close_file_err;

	const mx_size num_of_batches 	= size / batch_len;
	const mx_size batch_cells 	= width * height * batch_len;
	const mx_size x = (vertical) ? batch_cells : (mx_size) (width * height);
	const mx_size y = (vertical) ? 1 : batch_len;

	mx_t** list = malloc(sizeof(*list) * num_of_batches);
	if (list == NULL)
		goto close_file_err;
	mx_size n;
	for (n = 0; n < num_of_batches; ++n) {
		list[n] = mx_create(x, y);
		if (list[n] == NULL)
			goto free_all_err;
		for (mx_size i = 0; i < batch_cells; ++i) {
			uint8_t byte;
			if (fread(&byte, 1, U8_LEN, f) != U8_LEN)
				goto free_all_err;
			list[n]->arr[i] = (mx_type) byte / 255;
		}
	}
	fclose(f);
	return (mx_iterator) {.list = list, .curr = 0, .size = num_of_batches};

free_all_err:
	for (int j = n; j > -1; --j)
		mx_destroy(list[j]);
	free(list);
close_file_err:
	fclose(f);
	return empty_iterator;
}
