#include "read_mnist.h"

#include <stdint.h>
#include <stdio.h>

#include "mx_iterator.h"
#include "mx.h"

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
#define I32_LEN 32

mx_iterator
read_idx3(mx_t list[], const char *filename, mx_size batch_len, uint8_t vertical)
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
	    || fread(&width,	1, I32_LEN, f) != I32_LEN
	    || !width || !height || !size)
		goto err;


	reverse_bytes_int32(&size);
	reverse_bytes_int32(&height);
	reverse_bytes_int32(&width);
err:
	fclose(f);
	return empty_iterator;
}