#ifndef _DEF_MX_ITER_H_
#define _DEF_MX_ITER_H_

#include "mx.h"
#include "stdint.h"

typedef struct {
    mx_t **list;
    size_t curr;
    size_t size;
}
def_mx_iter_data_t;

mx_t* def_iter_next(void *iterator_data);
uint8_t def_iter_has_next(void *iterator_data);
void def_iter_reset(void *iterator_data);
void free_default_iterator(void **iterator);

#endif
