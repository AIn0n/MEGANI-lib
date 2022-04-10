#ifndef _IM2COL_H_
#define _IM2COL_H_
#include "mx.h"

void im2col(
	const mx_t *in,
	mx_t *out,
	const mx_size in_x,
	const mx_size in_y,
	const mx_size in_z,
	const mx_size batch,
	const mx_size krnl_x, const mx_size krnl_y);
#endif
