#include "im2col.h"
/*
void im2col(
	const mx_t *in,
	mx_t *out,
	const MX_SIZE in_x,
	const MX_SIZE in_y,
	const MX_SIZE in_z,
	const MX_SIZE batch,
	const MX_SIZE krnl_x,
	const MX_SIZE krnl_y
) {
	MX_TYPE *in_ptr = in->arr;
	const MX_SIZE in_elem_size = in_x * in_y * in_z;
	//const MX_SIZE krnl_size = krnl_x * krnl_y;
	//iterate thru every batch
	for (int b = batch; b--; in_ptr += in_elem_size) {
		for (MX_SIZE z = 0; z < in_z; ++z) {
			for (MX_SIZE ky = 0; ky < krnl_y; ++ky) {
				for (MX_SIZE kx = 0; kx < krnl_x; ++kx) {
					
				}
			}
		}
	}
}
*/