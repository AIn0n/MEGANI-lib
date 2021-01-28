#ifndef _UT_MACROS_H_
#define _UT_MACROS_H_

#include "stdio.h"
#include "stdlib.h"

#define TEST_START(num, func)  static int test##num(void) { printf("-- test %i -- (%s)\n", num, func);
#define TEST_END  puts("---- OK!"); return 0; }
#define TEST_IF(x) if(x) { printf("---- ERROR! (line %i)\n", __LINE__); return 1; }
#define TEST_SIZE(mx, xx, yy) if(mx->x != xx || mx->y != yy) { printf("---- ERROR!\n------ WRONG SIZE - %i, %i - expected - %i, %i (line %i)\n",mx->x, mx->y, xx, yy, __LINE__); return 1; }
#define TEST_ERR(x, y, err) if((x > (y + err)) || (x < (y - err))) { printf("---- ERROR!\nexpected %lf, get %lf (line %i)\n", y, x, __LINE__); return 1;}
#endif