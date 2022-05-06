#ifndef _MX_ITER_FUNCS_H_
#define _MX_ITER_FUNCS_H_

#include "mx.h"

typedef struct {
    mx_t* (* next)(void *);
    uint8_t (* has_next)(void *);
}
mx_iter_funcs;

#endif