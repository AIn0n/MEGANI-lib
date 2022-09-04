#include "def_mx_iter.h"

uint8_t
def_iter_has_next(const struct mx_iterator_t *iterator)
{
	const def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator->data;
	return iter->curr < iter->size;
}

mx_t*
def_iter_next(struct mx_iterator_t *iterator)
{
	def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator->data;
	return iter->list[iter->curr++];
}

void
def_iter_reset(struct mx_iterator_t *iterator)
{
	def_mx_iter_data_t *iter = (def_mx_iter_data_t *) iterator->data;
	iter->curr = 0;
}

void
free_default_iterator_data(struct mx_iterator_t *iterator)
{
	if (iterator->data == NULL)
		return;
	def_iter_reset(iterator);
	while(def_iter_has_next(iterator)) {
		mx_destroy(def_iter_next(iterator));
	}
	def_mx_iter_data_t *data = (def_mx_iter_data_t *) iterator->data;
	free(data->list);
	free(data);
}
