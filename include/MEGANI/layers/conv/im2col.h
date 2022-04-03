#ifndef _IM2COL_H_
#define _IM2COL_H_
#include "mx.h"

void im2col(
	const mx_t *in,
	mx_t *out,
	const MX_SIZE in_x,
	const MX_SIZE in_y,
	const MX_SIZE in_z,
	const MX_SIZE batch,
	const MX_SIZE krnl_x, const MX_SIZE krnl_y);
#endif
