#ifndef _BGD_H_
#define _BGD_H_

#include "mx.h"
#include "nn.h"

typedef struct {
    mx_type alpha;
}
bgd_data_t;

uint8_t add_batch_gradient_descent(nn_t *nn, mx_type alpha);

#endif
