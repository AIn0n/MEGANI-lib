#include "def_mx_iter.h"
#include "read_idx3.h"
#include "get_mnist_labels.h"
#include "nn.h"
#include "dense.h"
#include "bgd.h"
#include "types_wrappers.h"
#include <stdio.h>
#include <errno.h>
#include "convolution.h"
#include "flatten.h"
#include "image_layer_data.h"

#define BATCH_SIZE 5

int
hor_max_idx_cmp(const mx_t a, const mx_t b)
{
	int result = 0;
	for (mx_size y = 0; y < a.y; ++y) {
		int max_a = 0, max_b = 0;
		for (mx_size x = 0; x < a.x; ++x) {
			if (a.arr[x + y * a.x] > a.arr[max_a + y * a.x])
				max_a = x;
			if (b.arr[x + y * b.x] > b.arr[max_b + y * b.x])
				max_b = x;
		}
		result += (max_a == max_b);
	}
	return result;
}
int
main(void)
{
	struct mx_iterator_t
		input = read_idx3("mnist/train-images-idx3-ubyte",		BATCH_SIZE, 1),
		expected = get_mnist_labels("mnist/train-labels-idx1-ubyte",	BATCH_SIZE),
		test_input = read_idx3("mnist/t10k-images-idx3-ubyte",		BATCH_SIZE, 1),
		test_expected = get_mnist_labels("mnist/t10k-labels-idx1-ubyte",BATCH_SIZE);

	nn_t *network = nn_create(28 * 28, BATCH_SIZE);
	add_convolution_layer(network,
		(img_size_t) {.x = 28, .y = 28, .z = 1},
		(img_size_t) {.x = 4, .y = 4, .z = 16},
		2, RELU, -0.01, 0.01);
	add_flatten_layer(network);
	LAYER_DENSE(network, 10, NO_FUNC, -0.01, 0.01);
	add_batch_gradient_descent(network, 0.01);

	if (input.data == NULL || expected.data == NULL || test_input.data == NULL
	    || test_expected.data == NULL || network->error) {
		perror("some resources cannot be readed nor allocated");
		goto free_memory;
	}
	
	nn_fit_all(network, &input, &expected, 1);

	int n = 0, errors = 0;
	mx_t 	*expected_ptr	= test_expected.next(&test_expected),
		*test_input_ptr	= test_input.next(&test_input);

	while(test_expected.has_next(&test_expected) && test_input.has_next(&test_input)) {
		nn_predict(network, test_input_ptr);
		errors += hor_max_idx_cmp(*network->layers[network->len - 1].out, *expected_ptr);
		expected_ptr	= test_expected.next(&test_expected),
		test_input_ptr	= test_input.next(&test_input);
		++n;
	}
	printf("accuracy %.3lf%%\n", (double) (errors * 100) / (n * BATCH_SIZE));
free_memory:
	free_default_iterator_data(&input);
	free_default_iterator_data(&expected);
	free_default_iterator_data(&test_input);
	free_default_iterator_data(&test_expected);
	nn_destroy(network);
	return 0;
}
