#include "def_mx_iter.h"
#include "read_idx3.h"
#include "get_mnist_labels.h"
#include "nn.h"
#include "dense.h"
#include "bgd.h"
#include "types_wrappers.h"
#include <stdio.h>

#define BATCH_SIZE 10

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
	struct mx_iterator_t input = read_idx3("mnist/train-images-idx3-ubyte", BATCH_SIZE, 0);
	struct mx_iterator_t expected = get_mnist_labels("mnist/train-labels-idx1-ubyte", BATCH_SIZE);

	if (input.data == NULL || expected.data == NULL) {
		free_default_iterator_data(&input);
		free_default_iterator_data(&expected);
		return 1;
	}

	nn_t *network = nn_create(10, BATCH_SIZE);
	LAYER_DENSE(network, 300, RELU, -0.01, 0.01);
	LAYER_DENSE(network, 10, NO_FUNC, 0.0, 0.0);
	add_batch_gradient_descent(network, 0.01);
	if (network->error) {
		free_default_iterator_data(&input);
		free_default_iterator_data(&expected);
		nn_destroy(network);
		return 1;
	}
	
	nn_fit_all(network, &input, &expected, 1);
	free_default_iterator_data(&input);
	free_default_iterator_data(&expected);

	input = read_idx3("mnist/t10k-images-idx3-ubyte", BATCH_SIZE, 0);
	expected = get_mnist_labels("mnist/t10k-labels-idx1-ubyte", BATCH_SIZE);

	if (input.data == NULL || expected.data == NULL) {
		free_default_iterator_data(&input);
		free_default_iterator_data(&expected);
		nn_destroy(network);
		return 0;
	}

	int n = 0, errors = 0;
	mx_t 	*expected_ptr	= expected.next(&expected),
		*input_ptr	= input.next(&input);
	printf("%i\n",expected.has_next(&expected));
	while(expected.has_next(&expected) && input.has_next(&input)) { 	
		nn_predict(network, input_ptr);
		errors += hor_max_idx_cmp(*network->layers[network->len].out, *expected_ptr);

		expected_ptr	= expected.next(&expected),
		input_ptr	= input.next(&input);
	}

	printf("test error rate %.3lf%%\n", (double) (errors * 100) / n);
	free_default_iterator_data(&input);
	free_default_iterator_data(&expected);
	nn_destroy(network);
	return 0;
}
