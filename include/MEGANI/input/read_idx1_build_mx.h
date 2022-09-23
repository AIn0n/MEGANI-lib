#ifndef _READ_IDX1_BUILD_MX_H_
#define _READ_IDX1_BUILD_MX_H_

#include "mx_iterator.h"

struct mx_iterator_t read_idx1_build_mx(
	const char *filename,
	const mx_size batch,
	mx_t *(*build_mx)(mx_size, uint8_t *));

#endif
