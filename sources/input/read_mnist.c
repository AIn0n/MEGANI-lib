#include "read_mnist.h"

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

mx_iterator
read_idx3(const char *filename, mx_size batch_len, uint8_t vertical)
{
	FILE *f = fopen(filename, "rb");
	if (f == NULL || batch_len == 0)
		return empty_iterator;
	/* 
	* read whole file header build with magic number, size, height width
	* everything is done in single if to make this optimal and small
	* also, I think that it can be helpful for function caching but idk
	* am not cpu lol
	*/
	int32_t magic_num, size, height, width;
	if (fread(&magic_num,	1, I32_LEN, f) != I32_LEN
	    || fread(&size,	1, I32_LEN, f) != I32_LEN
	    || fread(&height,	1, I32_LEN, f) != I32_LEN
	    || fread(&width,	1, I32_LEN, f) != I32_LEN)
		goto err_close_file;

	reverse_bytes_int32(&magic_num);
	reverse_bytes_int32(&size);
	reverse_bytes_int32(&height);
	reverse_bytes_int32(&width);
	
	if (magic_num != 2051 || !width || !height || !size)
		goto err_close_file;

	const mx_size num_of_batches = size / batch_len;
	const mx_size batch_cells = width * height * batch_len;

	mx_t** list = malloc(sizeof(*list) * num_of_batches);
	if (list == NULL)
		goto err_close_file;

	const mx_size x = (vertical) ? batch_cells : (mx_size) (width * height);
	const mx_size y = (vertical) ? 1 : batch_len;

	uint8_t byte;
	mx_size n;
	for (n = 0; n < num_of_batches; ++n) {
		list[n] = mx_create(x, y);
		if (list[n] == NULL)
			goto free_all_err;
		for (mx_size i = 0; i < batch_cells; ++i) {
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
err_close_file:
	fclose(f);
	return empty_iterator;
}