#ifndef _BGD_H_
#define _BGD_H_

#include "mx.h"

typedef struct {
    mx_type alpha;
}
bgd_data_t;

void bgd_optimize(mx_t *vdelta, mx_t *weights, void *params);

#endif
