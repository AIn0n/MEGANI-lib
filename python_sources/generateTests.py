from testsGenerator import TestsGenerator, genAssert
import random

#	matrix create tests

gen = TestsGenerator()
for pair in [[0, 0], [1, 0], [0, 1]]:
	gen.genTest('mx_create',
		'\tmx_t *a = mx_create({}, {});\n'.format(*pair) +
		genAssert('a != NULL'))

x = random.randrange(1, 255)
y = random.randrange(1, 255)
gen.genTest('mx_create',
	f'\tmx_t *a = mx_create({x}, {y});\n' +
	genAssert('a == NULL') + 
	genAssert(f'a->x != {x} || a->y != {y}'))

# matrix multiplication tests



gen.save('tests.c')