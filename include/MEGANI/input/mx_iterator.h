#ifndef _MX_ITER_FUNCS_H_
#define _MX_ITER_FUNCS_H_

#include "mx.h"

struct mx_iterator_t {
    mx_t* (* next)(struct mx_iterator_t *);
    uint8_t (* has_next)(const struct mx_iterator_t *);
    void (* reset)(struct mx_iterator_t *);
    void *data;
};

#endif
