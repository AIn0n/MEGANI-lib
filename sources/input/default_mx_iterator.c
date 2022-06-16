#include "def_mx_iter.h"

uint8_t
def_iter_has_next(void *iterator)
{
	const def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator;
	return iter->curr < iter->size;
}

mx_t*
def_iter_next(void *iterator)
{
	def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator;
	return iter->list[iter->curr++];
}

void
def_iter_reset(void *iterator)
{
	def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator;
	iter->curr = 0;
}

void
free_default_iterator(void **iterator)
{
	def_mx_iter_data_t *iter = (def_mx_iter_data_t *) *iterator;
        do {
                mx_destroy(def_iter_next(iter));
        } 
	while(def_iter_has_next(iter));
        free(iter->list);
	free(*iterator);
}
