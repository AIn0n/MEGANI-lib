# 	---=== predefined strings ===---

_TEST_ASSERT = """\tif ({}) {{\n\t\tprintf("---- ERROR! (line %i)\\n", __LINE__);\n\t\treturn 1;\n\t}}\n"""
_TEST_START = """\nstatic int\ntest{number}(void)\n{{\n\tputs("-- TEST {number} -- {name}");\n\n"""
_TEST_END = """\tputs("---- OK!");\n\treturn 0;\n}\n"""
_START_MAIN = """\nint\nmain(void) {\n\tint (* test_funcs[]) (void) = {"""

_INCLUDE = """#include <stdio.h>
#include "mx.h"
#include "nn.h"
#include "dense.h"
#include "mx_iterator.h"
"""

_END_MAIN = """}};
	const int test_size = {numOfTests};
	int failed = 0;
	for (int i = 0; i < test_size; ++i)
		failed += (*test_funcs[i])();
	
	printf("%i of %i tests failed\\n", failed, test_size);
	return failed;\n}}"""

# 	---=== generators of static declarations ===---

# generate static matrix declaration
def genStaticMxDec(mx, mxName: str) -> str:
    return (
        f"""	mx_t {mxName} = {{.x = {mx.shape[1]}, .y = {mx.shape[0]}, .size = {mx.size}}};
	mx_type {mxName}_arr[] = {{\n"""
        + "".join("\t\t" + "".join(f"{x}, " for x in y) + "\n" for y in mx)
        + f"\n\t}};\n\t{mxName}.arr = {mxName}_arr;\n\n"
    )


# generate static empty matrix declaration
def genStaticEmptyMxDec(shape, mxName: str) -> str:
    size = shape[0] * shape[1]
    return (
        f"""	mx_t {mxName} = {{.x = {shape[1]}, .y = {shape[0]}, .size = {size}}};
	mx_type {mxName}_arr[{size}];"""
        + f"\n\t{mxName}.arr = {mxName}_arr;\n\n"
    )


# generate static list declaration
def genStaticListDec(l, lName: str) -> str:
    return f"\tmx_type {lName}[] = {{" + "".join(f"{n}, " for n in l) + "};\n\n"


# 	---=== generators of macro-like functions ===---

# generate matrix comparison
# 	arguments:
# 	* mx - matrix name as string
# 	* l  - list name as string
# 	* delta - max acceptable difference between expected and got value
def genMxComp(mx: str, l: str, delta: float) -> str:
    return f"""	for (mx_size i = 0; i < {mx}.size; ++i)
		if ({mx}.arr[i] > {l}[i] + {delta} || {mx}.arr[i] < {l}[i] - {delta}) {{
			printf("---- ERROR! (line %i)\\n------ expected -> %f, got -> %f\\n", __LINE__, {l}[i], {mx}.arr[i]);
			return 1;
		}}
"""


# generate matrix/list copy
# 	arguments:
# 	* src - source list name as string
# 	* dest - destination list name as string
# 	* size - size as string
def genListCpy(src: str, dest: str, size: str) -> str:
    return f"""	for (mx_size i = 0; i < {size}; ++i)
		{dest}[i] = {src}[i];
"""


# generate static if-like assertion
def genAssert(code: str) -> str:
    return _TEST_ASSERT.format(code)


class TestsGenerator:
    def __init__(self) -> None:
        self.testCounter = 0
        self.code = _INCLUDE

    def genTest(self, testName: str, code: str) -> None:
        self.code += (
            _TEST_START.format(number=self.testCounter, name=testName)
            + code
            + _TEST_END
        )
        self.testCounter += 1

    def appendCode(self, code: str) -> None:
        self.code += code

    def __genMain(self) -> None:
        self.code += _START_MAIN
        self.code += "".join(f"test{n}," for n in range(self.testCounter))
        self.code += _END_MAIN.format(numOfTests=self.testCounter)

    def save(self, filepath) -> None:
        self.__genMain()
        with open(filepath, "w") as f:
            print(self.code, file=f)
