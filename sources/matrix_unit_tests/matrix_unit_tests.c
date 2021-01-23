#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"

#define TEST_START(num, func) int test##num(void) { printf("-- test %i -- (%s)\n", num, func);
#define TEST_END  puts("---- OK!"); return 0; }
#define TEST_IF(x) if(x) { printf("---- ERROR! (line %i)\n", __LINE__); return 1; }

TEST_START(1, "matrix_create")
{
    matrix_t *a = matrix_create(0, 0);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(2, "matrix_create")
{
    matrix_t *a = matrix_create(1, 0);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(3, "matrix_create")
{
    matrix_t *a = matrix_create(0, 1);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(4, "matrix_create")
{
    matrix_t *a = matrix_create(1, 1);
    TEST_IF(a == NULL)
    TEST_IF(a->x != 1)
    TEST_IF(a->y != 1)
    matrix_free(a);
}
TEST_END

int main (void) 
{ 
    int (*test_ptr_arr[])(void) = { test1, test2, test3 };
    const int test_size = sizeof(test_ptr_arr)/sizeof(test_ptr_arr[0]);
    int failed = 0;

    for(int i = 0; i < test_size; ++i) failed += (*test_ptr_arr[i])();
    
    printf("%i of %i tests failed\n", failed, test_size);
    return 0; 
}