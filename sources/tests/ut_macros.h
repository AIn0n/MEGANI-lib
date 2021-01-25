#ifndef _UT_MACROS_H_
#define _UT_MACROS_H_

#define TEST_START(num, func) int test##num(void) { printf("-- test %i -- (%s)\n", num, func);
#define TEST_END  puts("---- OK!"); return 0; }
#define TEST_IF(x) if(x) { printf("---- ERROR! (line %i)\n", __LINE__); return 1; }

#endif