from testsGenerator import *
import numpy as np
import random

#	matrix create tests

gen = TestsGenerator()
for pair in [[0, 0], [1, 0], [0, 1]]:
	gen.genTest('mx_create',
		'\tmx_t *a = mx_create({}, {});\n'.format(*pair) +
		genAssert('a != NULL'))

x, y = random.randint(1, 16), random.randint(1, 16)
gen.genTest('mx_create',
	f'\tmx_t *a = mx_create({x}, {y});\n' +
	genAssert('a == NULL') + 
	genAssert(f'a->x != {x} || a->y != {y} || a->size = {x * y}') +
	'\tmx_destroy(a);')

# matrix multiplication tests

for n in range(5):
	x1, y1 = random.randint(1, 16), random.randint(1, 16)
	x2 = random.randint(1, 16)
	mx_a = np.random.randint(254, size = (y1, x1))
	mx_b = np.random.randint(254, size = (x1, x2))
	expected = np.matmul(mx_a, mx_b).flatten()
	gen.genTest('mx_mp',
		genStaticMxDec(mx_a, 'a') +
		genStaticMxDec(mx_b, 'b') +
		genStaticEmptyMxDec((y1, x2), 'res') +
		genStaticListDec(expected, 'exp') +
		'\tmx_mp(a, b, &res, 0);\n\n' +
		genMxComp('res', 'exp')
	)

for n in range(5):
	x1, y1 = random.randint(1, 16), random.randint(1, 16)
	x2 = random.randint(1, 16)
	mx_a = np.random.randint(254, size = (y1, x1))
	mx_b = np.random.randint(254, size = (y1, x2))
	expected = np.matmul(np.transpose(mx_a), mx_b).flatten()
	gen.genTest('mx_mp',
		genStaticMxDec(mx_a, 'a') +
		genStaticMxDec(mx_b, 'b') +
		genStaticEmptyMxDec((x2, y1), 'res') +
		genStaticListDec(expected, 'exp') +
		'\tmx_mp(a, b, &res, A);\n\n' +
		genMxComp('res', 'exp')
	)

gen.save('tests.c')