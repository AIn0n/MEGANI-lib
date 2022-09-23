#ifndef _GET_MNIST_LABELS_H_
#define _GET_MNIST_LABELS_H_

#include "mx.h"

struct mx_iterator_t get_mnist_labels(const char *filename, const mx_size batch);

#endif
