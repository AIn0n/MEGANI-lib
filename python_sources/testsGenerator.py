_TEST_ASSERT =\
'''	if ({}) {{
		printf("---- ERROR! (line %i)\\n", __LINE__);
		return 1;
	}}
'''

_INCLUDE='''#include <stdio.h>
#include "mx.h"
'''

_TEST_START ='''
static int
test{number}(void) 
{{
	puts("-- TEST {number} -- {name}");
'''

_TEST_END =\
'''	puts("---- OK!");
	return 0;
}
'''

_START_MAIN ='''
int
main(void) {
	int (* test_funcs[]) (void) = {
'''

_END_MAIN ='''	}};
	const int test_size = {numOfTests};
	int failed = 0;
	for (int i = 0; i < test_size; ++i)
		failed += (*test_funcs[i])();
	
	printf("%i of %i tests failed\\n", failed, test_size);
	return failed;
}}'''

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

	def __genMain(self) -> None:
		self.code += _START_MAIN
		self.code += ''.join(
			f'\t\ttest{n},\n' for n in range(self.testCounter))
		self.code += _END_MAIN.format(numOfTests = self.testCounter)

	def save(self, filepath) -> None:
		self.__genMain()
		with open(filepath, 'w') as f:
			print(self.code, file = f)