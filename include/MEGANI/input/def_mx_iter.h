#ifndef _DEF_MX_ITER_H_
#define _DEF_MX_ITER_H_

#include "mx.h"
#include "stdint.h"
#include "mx_iterator.h"

typedef struct {
	mx_t **list;
	size_t curr;
	size_t size;
} def_mx_iter_data_t;

mx_t *def_iter_next(struct mx_iterator_t *iterator);
uint8_t def_iter_has_next(const struct mx_iterator_t *iterator);
void def_iter_reset(struct mx_iterator_t *iterator);
void free_default_iterator_data(struct mx_iterator_t *iterator);

#endif
