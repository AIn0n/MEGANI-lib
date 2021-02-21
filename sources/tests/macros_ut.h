#ifndef _UT_MACROS_H_
#define _UT_MACROS_H_

#include "stdio.h"
#include "stdlib.h"
#include "stdint.h"

#define TEST_START(num, func)  static int test##num(void) { printf("-- test %i -- (%s)\n", num, func);
#define TEST_END  puts("---- OK!"); return 0; }
#define TEST_IF(x) if(x) { printf("---- ERROR! (line %i)\n", __LINE__); return 1; }
#define TEST_SIZE(mx, xx, yy) if(mx->x != xx || mx->y != yy) { printf("---- ERROR!\n------ WRONG SIZE - %i, %i - expected - %i, %i (line %i)\n",mx->x, mx->y, xx, yy, __LINE__); return 1; }
#define TEST_ERR(x, y, err) if((x > (y + err)) || (x < (y - err))) { printf("---- ERROR!\nexpected %lf, get %lf (line %i)\n", y, x, __LINE__); return 1;}

#define MX_CMP(mx, array) do{ for(uint32_t i = 0; i < (mx.x * mx.y); ++i) if(mx.arr[i] != array[i]) {printf("---- ERROR!\n------ WRONG VALUE IN MATRIX - %lf - expected - %lf (line %i)\n", mx.arr[i], array[i], __LINE__); return 1;}}while(0);
#define MX_CPY(src, dir) do{ for(uint32_t i = 0; i < (dir->x * dir->y); ++i) dir->arr[i] = src[i]; }while(0);

#endif