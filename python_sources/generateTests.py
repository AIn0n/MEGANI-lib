from numpy.random.mtrand import rand, randint
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
	genAssert(f'a->x != {x} || a->y != {y} || a->size != {x * y}') +
	'\tmx_destroy(a);')

#	matrix multiplication tests

for t in ('0', 'A', 'B', 'BOTH'):
	for n in range(5):
		x1, y1 = random.randint(1, 16), random.randint(1, 16)
		x2, y2 = {
			'0'	: (random.randint(1, 16), x1), 
			'A'	: (random.randint(1, 16), y1),
			'B' 	: (x1, random.randint(1, 16)), 
			'BOTH'	: (y1, random.randint(1, 16))}[t]

		x3, y3 = {'0':(x2,y1),'A':(x2,x1),'B':(y2,y1),'BOTH':(y2,x1)}[t]
		a = np.random.randint(-254, high = 254, size = (y1, x1))
		b = np.random.randint(-254, high = 254, size = (y2, x2))
		expected = np.matmul(
			np.transpose(a) if t in ('A', 'BOTH') else a, 
			np.transpose(b) if t in ('B', 'BOTH') else b).flatten()

		gen.genTest('mx_mp' if t == '0' else 'mx_mp (transposed)',
			genStaticMxDec(a, 'a') +
			genStaticMxDec(b, 'b') +
			genStaticEmptyMxDec((y3, x3), 'res') +
			genStaticListDec(expected, 'exp') +
			f'\tmx_mp(a, b, &res, {t});\n\n' +
			genMxComp('res', 'exp')
		)

#	matrix hadamard product tests

for n in range(5):
	x, y = random.randint(1, 16), random.randint(1, 16)
	a = np.random.randint(-254, high = 256, size = (y, x))
	b = np.random.randint(-254, high = 256, size = (y, x))
	expected = np.multiply(a, b).flatten()
	gen.genTest('mx_hadamard',
		genStaticMxDec(a, 'a') +
		genStaticMxDec(b, 'b') +
		genStaticEmptyMxDec((y, x), 'res') +
		genStaticListDec(expected, 'exp') +
		f'\tmx_hadamard(a, b, &res);\n' +
		genMxComp('res', 'exp')
	)

#	matrix substraction

for n in range(5):
	x, y = random.randint(1, 16), random.randint(1, 16)
	a = np.random.randint(-254, high = 254, size = (y, x))
	b = np.random.randint(-254, high = 254, size = (y, x))
	expected = (a - b).flatten()
	gen.genTest('mx_substraction',
		genStaticMxDec(a, 'a') +
		genStaticMxDec(b, 'b') +
		genStaticEmptyMxDec((y, x), 'res') +
		genStaticListDec(expected, 'exp') +
		f'\tmx_sub(a, b, &res);\n' +
		genMxComp('res', 'exp')
	)

#	matrix multiply by number

for n in range(5):
	x, y = random.randint(1, 16), random.randint(1, 16)
	a = np.random.randint(-254, high = 254, size = (y, x))
	num = random.randint(-16, 16)
	expected = (a * num).flatten()
	gen.genTest('mx_mp_num',
		genStaticMxDec(a, 'a') +
		genStaticListDec(expected, 'exp') +
		f'\tmx_mp_num(&a, {num});\n' +
		genMxComp('a', 'exp')
	)

#	matrix hadamard product with lambda

gen.appendCode('\nMX_TYPE foo(MX_TYPE a) {return (a > 4) ? 0 : a;}\n')

x, y = random.randint(1, 16), random.randint(1, 16)
a = np.random.randint(-254, high = 254, size = (y, x))
b = np.random.randint(-254, high = 254, size = (y, x))
new_b = np.array([[0 if x > 4 else x for x in y] for y in b])
expected = (a * new_b).flatten()

gen.genTest('mx_hadam_lambda',
	genStaticMxDec(a, 'a') +
	genStaticMxDec(b, 'b') +
	f'\tmx_hadam_lambda(&a, b, (&foo));\n' +
	genStaticListDec(expected, 'exp') +
	genMxComp('a', 'exp')
)
gen.save('sources/tests/main.c')