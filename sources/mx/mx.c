#include "MEGANI/mx/mx.h" /* matrix type, and types related to matrices */
#include <stdio.h> //FOR DEBUG ONLY

inline void
mx_set_size(mx_t *mx, const mx_size x, const mx_size y)
{
	mx->x = x;
	mx->y = y;
	mx->size = x * y;
}

mx_t *
mx_create(const mx_size x, const mx_size y)
{
	if (!x || !y)
		return NULL;

	mx_t *output = calloc(1, sizeof(*output));
	if (output == NULL)
		return NULL;
	mx_set_size(output, x, y);
	if ((output->arr = calloc(output->size, sizeof(*output->arr))) ==
	    NULL) {
		free(output);
		return NULL;
	}
	return output;
}

int
mx_hor_max_idx_cmp(const mx_t a, const mx_t b)
{
	int result = 0;
	for (mx_size y = 0; y < a.y; ++y) {
		int max_a = 0, max_b = 0;
		for (mx_size x = 0; x < a.x; ++x) {
			if (a.arr[x + y * a.x] > a.arr[max_a + y * a.x])
				max_a = x;
			if (b.arr[x + y * b.x] > b.arr[max_b + y * b.x])
				max_b = x;
		}
		result += (max_a == max_b);
	}
	return result;
}

uint8_t
mx_recreate(mx_t *mx, const mx_size x, const mx_size y)
{
	mx_type *new_arr = realloc(mx->arr, x * y * sizeof(*new_arr));
	if (new_arr == NULL)
		return 1;
	mx->arr = new_arr;
	mx_set_size(mx, x, y);
	return 0;
}

uint8_t
mx_recreate_if_too_small(mx_t *mx, const mx_size x, const mx_size y)
{
	if (mx->size < x * y)
		return mx_recreate(mx, x, y);
	return 0;
}

void
mx_fill_rng(mx_t *values, const mx_type min, const mx_type max)
{
	const mx_type diff = (max - min);
	for (mx_size i = 0; i < values->size; ++i) {
		mx_type rand_val = (mx_type)rand() / RAND_MAX;
		values->arr[i] = min + rand_val * diff;
	}
}

void
mx_elem_power_by_two(mx_t *mx)
{
	mx_type *addr = mx->arr;
	for (mx_size n = mx->size; n > 0; --n, ++addr)
		*addr *= *addr;
}

void
mx_destroy(mx_t *mx)
{
	if (mx == NULL)
		return;
	free(mx->arr);
	free(mx);
	mx = NULL;
}

void
mx_mp(const mx_t a, const mx_t b, mx_t *out, const mx_mp_params params)
{
	mx_size stride_ya = a.x, stride_xa = 1, limit = a.x;
	if (params & A) {
		limit = a.y;
		stride_ya = 1;
		stride_xa = a.x;
	}
	mx_size stride_xb = 1, stride_yb = b.x;
	if (params & B) {
		stride_xb = b.x;
		stride_yb = 1;
	}

	mx_type *addrC = out->arr;
	for (mx_size n = out->size; n > 0; --n, ++addrC)
		*addrC = 0;
	const mx_type *addrA = a.arr, *addrB = b.arr;
	for (mx_size i = 0; i < limit; ++i) {
		addrC = out->arr;
		for (mx_size y = 0; y < out->y; ++y) {
			for (mx_size x = 0; x < out->x; ++x)
				addrC[x] += addrA[y * stride_ya] *
					    addrB[x * stride_xb];
			addrC += out->x;
		}
		addrA += stride_xa;
		addrB += stride_yb;
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
mx_add_to_first(mx_t *a, const mx_t *b)
{
	for (mx_size n = 0; n < a->size; ++n)
		a->arr[n] += b->arr[n];
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
