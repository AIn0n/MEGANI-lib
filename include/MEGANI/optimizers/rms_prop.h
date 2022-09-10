#ifndef _RMS_PROP_H_
#define _RMS_PROP_H_

#include "nn.h"

typedef struct {
    mx_type alpha;
    mx_type rho;
    mx_t **caches;
}
rms_prop_data_t;

optimizer_t rms_prop_create(nn_t *nn, mx_type alpha, mx_type rho);
void rms_prop_destroy(optimizer_t self);

#endif
