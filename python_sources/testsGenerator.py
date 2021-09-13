_TEST_ASSERT	= '''\tif ({}) {{\n\t\tprintf("---- ERROR! (line %i)\\n", __LINE__);\n\t\treturn 1;\n\t}}\n'''
_INCLUDE	= '''#include <stdio.h>\n#include "mx.h"\n'''
_TEST_START 	= '''\nstatic int\ntest{number}(void)\n{{\n\tputs("-- TEST {number} -- {name}");\n\n'''
_TEST_END	= '''\tputs("---- OK!");\n\treturn 0;\n}\n'''
_START_MAIN	= '''\nint\nmain(void) {\n\tint (* test_funcs[]) (void) = {'''

_END_MAIN ='''}};
	const int test_size = {numOfTests};
	int failed = 0;
	for (int i = 0; i < test_size; ++i)
		failed += (*test_funcs[i])();
	
	printf("%i of %i tests failed\\n", failed, test_size);
	return failed;\n}}'''

def genStaticMxDec(mx, mxName :str) -> str:
	return \
f'''	mx_t {mxName} = {{.x = {mx.shape[1]}, .y = {mx.shape[0]}, .size = {mx.size}}};
	MX_TYPE {mxName}_arr[] = {{\n''' +\
	''.join('\t\t' + ''.join(f'{x}, ' for x in y) + '\n' for y in mx) +\
	f'\n\t}};\n\t{mxName}.arr = {mxName}_arr;\n\n'

def genStaticEmptyMxDec(shape, mxName :str) -> str:
	size = shape[0] * shape[1]
	return \
f'''	mx_t {mxName} = {{.x = {shape[1]}, .y = {shape[0]}, .size = {size}}};
	MX_TYPE {mxName}_arr[{size}];''' +\
	f'\n\t{mxName}.arr = {mxName}_arr;\n\n'

def genMxComp(mxName :str, l :str) -> str:
	return \
f'''	for (MX_SIZE n = 0; n < {mxName}.size; ++n)
		if ({mxName}.arr[n] != {l}[n]) {{
			printf("---- ERROR! (line %i)\\n------ expected -> %f, got -> %f\\n", __LINE__, {l}[n], {mxName}.arr[n]);
			return 1;
		}}
'''

def genStaticListDec(l, lName :str) -> str:
	return \
	f'\tMX_TYPE {lName}[] = {{' + ''.join(f'{n}, ' for n in l) + '};\n\n'

def genAssert(code :str) -> str:
	return _TEST_ASSERT.format(code)

class TestsGenerator:
	def __init__(self) -> None:
	    self.testCounter = 0
	    self.code = _INCLUDE
	
	def genTest(self, testName :str, code :str) -> None:
		self.code +=\
		_TEST_START.format(number = self.testCounter, name = testName) +\
		code + _TEST_END
		self.testCounter += 1

	def appendCode(self, code :str) -> None:
		self.code += code

	def __genMain(self) -> None:
		self.code += _START_MAIN
		self.code += ''.join(
			f'test{n},' for n in range(self.testCounter))
		self.code += _END_MAIN.format(numOfTests = self.testCounter)

	def save(self, filepath) -> None:
		self.__genMain()
		with open(filepath, 'w') as f:
			print(self.code, file = f)