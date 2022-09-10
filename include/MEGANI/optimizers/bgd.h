#ifndef _BGD_H_
#define _BGD_H_

#include "nn.h"

typedef struct {
    mx_type alpha;
}
bgd_data_t;

optimizer_t bgd_create(nn_t *nn, mx_type alpha);
void bgd_destroy(optimizer_t self);

#endif
