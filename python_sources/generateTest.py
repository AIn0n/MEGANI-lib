test_counter = 0

INCLUDE='''#include <stdio.h>
'''

TEST_START =\
'''
static int
test{number}(void) 
{{
	puts("-- TEST {number} -- {name}");
'''

TEST_ASSERT =\
'''	if ({}) {{
		printf("---- ERROR! (line %i)\\n", __LINE__);
		return 1;
	}}
'''

TEST_END =\
'''	puts("---- OK!");
	return 0;
}
'''

START_MAIN ='''
int
main(void) {
	int (* test_funcs[]) (void) = {
'''

END_MAIN ='''	}};
	const int test_size = {numOfTests};
	int failed = 0;
	for (int i = 0; i < test_size; ++i)
		failed += (*test_funcs[i])();
	
	printf("%i of %i tests failed\\n", failed, test_size);
	return failed;
}}'''

class TestsGen:
	def __init__(self) -> None:
	    self.testCounter = 0
	    self.code = ''.join(INCLUDE)
	
	def genTest(self, testName :str, code :str) -> None:
		self.code +=\
		TEST_START.format(number = self.testCounter, name = testName) +\
		code + TEST_END
		self.testCounter += 1

	def __genMain(self):
		self.code += START_MAIN
		self.code += ''.join(
			f'''\t\ttest{n},\n''' for n in range(self.testCounter))
		self.code += END_MAIN.format(numOfTests = self.testCounter)

	def save(self, filepath):
		self.__genMain()
		with open(filepath, 'w') as f:
			print(self.code, file = f)

tests = TestsGen()
tests.genTest('test', TEST_ASSERT.format('x == 0'))
tests.genTest('yet another tests',
'''	int y = 0;
''' + TEST_ASSERT.format("y != 0"))
tests.save('tests.c')