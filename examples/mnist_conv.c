#include <stdio.h>

#include "def_mx_iter.h" /* default iterator operations */
#include "read_idx3.h" /* read mnist images */
#include "get_mnist_labels.h" /* read mnist labels */
#include "nn.h" /* basic nerual netwok type and functions */
#include "dense.h" /* dense layer */
#include "bgd.h" /* batch gradient descent optimizer */
#include "convolution.h" /* convolution layer */
#include "flatten.h" /* flatten layer */
#include "image_layer_data.h" /* img_size_t type structure used in convolution creation */

#define BATCH_SIZE 5

int main(void)
{
	struct mx_iterator_t
		input = read_idx3("mnist/train-images-idx3-ubyte",		BATCH_SIZE, 1),
		expected = get_mnist_labels("mnist/train-labels-idx1-ubyte",	BATCH_SIZE),
		test_input = read_idx3("mnist/t10k-images-idx3-ubyte",		BATCH_SIZE, 1),
		test_expected = get_mnist_labels("mnist/t10k-labels-idx1-ubyte",BATCH_SIZE);

	nn_t *network = nn_create(28 * 28, BATCH_SIZE);
	add_convolution_layer(
		network,
		(img_size_t){ .x = 28, .y = 28, .z = 1 }, /* input dims */
		(img_size_t){ .x = 4, .y = 4, .z = 16 }, /* kernel 4x4, 16 channels */
		2, RELU, -0.01, 0.01); /* stride = 2 */
	add_flatten_layer(network);
	LAYER_DENSE(network, 10, NO_FUNC, -0.01, 0.01);
	optimizer_t optimizer = bgd_create(network, 0.01);

	if (input.data == NULL || expected.data == NULL ||
	    test_input.data == NULL || test_expected.data == NULL ||
	    network->error || optimizer.size < 1) {
		printf("Some resources cannot be allocated nor created");
		goto free_memory;
	}

	nn_fit_all(network, optimizer, &input, &expected, 1);

	int n = 0, errors = 0;
	mx_t *expected_ptr = test_expected.next(&test_expected),
	     *test_input_ptr = test_input.next(&test_input);

	while (test_expected.has_next(&test_expected) &&
	       test_input.has_next(&test_input)) {
		nn_predict(network, test_input_ptr);
		errors += mx_hor_max_idx_cmp(
			*network->layers[network->len - 1].out, *expected_ptr);
		expected_ptr = test_expected.next(&test_expected),
		test_input_ptr = test_input.next(&test_input);
		++n;
	}
	printf("accuracy %.3lf%%\n", (double)(errors * 100) / (n * BATCH_SIZE));
free_memory:
	free_default_iterator_data(&input);
	free_default_iterator_data(&expected);
	free_default_iterator_data(&test_input);
	free_default_iterator_data(&test_expected);
	nn_destroy(network);
	bgd_destroy(optimizer);
	return 0;
}
