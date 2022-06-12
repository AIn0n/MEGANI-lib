#ifndef _RMS_PROP_H_
#define _RMS_PROP_H_

#include "nn.h"

typedef struct {
    mx_type alpha;
    mx_type rho;
    uint32_t iterations;
    mx_t **caches;
}
rms_prop_data_t;

uint8_t add_rms_prop(nn_t *nn, mx_type alpha, mx_type rho);

#endif
