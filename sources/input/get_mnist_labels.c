#include "mx.h"
#include "read_idx1_build_mx.h"

#define NUM_OF_LABELS 10

static mx_t*
build_mnist_labels(mx_size size, uint8_t *buff)
{
	mx_t *result = mx_create(NUM_OF_LABELS, size);
	if (result != NULL)
		for (mx_size y = 0; y < size; ++y)
			for (mx_size x = 0; x < NUM_OF_LABELS; ++x)
				result->arr[x + y * result->x] = (buff[y] == x) ? 1 : 0;
	return result;
}

struct mx_iterator_t
get_mnist_labels(const char *filename, const mx_size batch)
{
	return read_idx1_build_mx(filename, batch, build_mnist_labels);
}
