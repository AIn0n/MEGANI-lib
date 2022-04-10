#include "mx.h"
#include <stdio.h>  //FOR DEBUG ONLY

inline void
mx_set_size(mx_t *mx, const mx_size x, const mx_size y)
{
	mx->x = x;
	mx->y = y;
	mx->size = x * y;
}

mx_t* 
mx_create(const mx_size x, const mx_size y)
{
	if (!x || !y) 
		return NULL;

	mx_t *output = calloc(1, sizeof(*output));
	if (output == NULL)
		return NULL;
	mx_set_size(output, x, y);
	output->arr = calloc(output->size, sizeof(*output->arr));
	if (output->arr == NULL) { 
		free(output);
		return NULL;
	}
	return output;
}

void
mx_destroy(mx_t *mx)
{
	if (mx == NULL)
		return;
	if (mx->arr != NULL) 
		free(mx->arr);
	free(mx);
}

void 
mx_mp(const mx_t a, const mx_t b, mx_t *out, const mx_mp_params params)
{
	mx_size tra_y = a.x, tra_i = 1, limit = a.x;
	if (params & A) {
		tra_y = 1;
		tra_i = a.x;
		limit = a.y;
	}
	mx_size trb_x = 1, trb_i = b.x;
	if (params & B) {
		trb_x = b.x;
		trb_i = 1;
	}
	for (mx_size y = 0; y < out->y; ++y) {
		for (mx_size x = 0; x < out->x; ++x) {
			mx_type val = 0;
			for (mx_size i = 0; i < limit; ++i) {
				val +=	a.arr[i * tra_i + y * tra_y] * 
					b.arr[x * trb_x + i * trb_i];
			}
			out->arr[x + y * out->x] = val; 
		}
	}
}

void 
mx_hadamard(const mx_t a, const mx_t b, mx_t *out)
{
	for (mx_size i = 0; i < out->size; ++i)
		out->arr[i] = a.arr[i] * b.arr[i];
}

void
mx_sub(const mx_t a, const mx_t b, mx_t *out)
{
	for (mx_size i = 0; i < out->size; ++i)
		out->arr[i] = a.arr[i] - b.arr[i];
}

void 
mx_mp_num(mx_t *a, const mx_type num)
{
	for (mx_size i = 0; i < a->size; ++i)
		a->arr[i] *= num;
}

void 
mx_hadam_lambda(mx_t *a, const mx_t b, mx_type (*lambda)(mx_type))
{
	for (mx_size i = 0; i < a->size; ++i)
		a->arr[i] *= (*lambda)(b.arr[i]);
}

//---------------------------------DEBUG ONLY-----------------------------------

void
mx_print(const mx_t *a, char *name)
{
	printf("%s\n", name);
	if (a == NULL) {
		puts("NULL");
		return;
	}
	for (mx_size i = 0; i < a->x * a->y; ++i) {
		if(i && !(i % a->x)) 
			puts("");
		printf("%lf ", a->arr[i]);
	}
	puts("");
}
