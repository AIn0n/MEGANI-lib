#include "mx_iterator.h"

uint8_t
default_iter_has_next(void* iterator)
{
	const mx_iterator *iter = (mx_iterator *) iterator;
	return iter->curr < iter->size;
}

mx_t*
default_iter_next(void* iterator)
{
	mx_iterator *iter = (mx_iterator *) iterator;
	return iter->list[iter->curr++];
}
