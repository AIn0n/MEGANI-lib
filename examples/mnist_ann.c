#include <stdio.h>

#include "def_mx_iter.h"	/* default iterator operations */
#include "read_idx3.h"		/* read mnist images */
#include "get_mnist_labels.h"	/* read mnist labels */
#include "nn.h"			/* basic nerual netwok type and functions */
#include "dense.h"		/* dense layer */
#include "bgd.h"		/* batch gradient descent optimizer */

#define BATCH_SIZE 50

int
main(void)
{
	struct mx_iterator_t
		input = read_idx3("mnist/train-images-idx3-ubyte",		BATCH_SIZE, 0),
		expected = get_mnist_labels("mnist/train-labels-idx1-ubyte",	BATCH_SIZE),
		test_input = read_idx3("mnist/t10k-images-idx3-ubyte",		BATCH_SIZE, 0),
		test_expected = get_mnist_labels("mnist/t10k-labels-idx1-ubyte",BATCH_SIZE);

	nn_t *network = nn_create(28 * 28, BATCH_SIZE);
	LAYER_DENSE(network, 32, RELU, -0.01, 0.01);
	LAYER_DENSE(network, 10, NO_FUNC, -0.01, 0.01);
	add_batch_gradient_descent(network, 0.01);

	if (input.data == NULL || expected.data == NULL || test_input.data == NULL
	    || test_expected.data == NULL || network->error) {
		puts("some resources cannot be readed nor allocated");
		goto free_memory;
	}
	
	nn_fit_all(network, &input, &expected, 2);

	int n = 0, errors = 0;
	mx_t 	*expected_ptr	= test_expected.next(&test_expected),
		*test_input_ptr	= test_input.next(&test_input);

	while(test_expected.has_next(&test_expected) && test_input.has_next(&test_input)) {
		nn_predict(network, test_input_ptr);
		errors += mx_hor_max_idx_cmp(*network->layers[network->len - 1].out, *expected_ptr);
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
