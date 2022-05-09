#ifndef _MX_ITERATOR_H_
#define _MX_ITERATOR_H_

#include "mx.h"
#include "stdint.h"

typedef struct {
    mx_t** list;
    size_t curr;
    size_t size;
}
mx_iterator;

mx_t* default_iter_next(void* iterator);
uint8_t default_iter_has_next(void* iterator);

#endif
