#include "matrix.h"
#include "stdio.h"
#include "stdlib.h"

#define TEST_START(num, func) int test##num(void) { printf("-- test %i -- (%s)\n", num, func);
#define TEST_END  puts("---- OK!"); return 0; }
#define TEST_IF(x) if(x) { printf("---- ERROR! (line %i)\n", __LINE__); return 1; }

TEST_START(1, "mx_create")
{
    mx_t *a = mx_create(0, 0);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(2, "mx_create")
{
    mx_t *a = mx_create(1, 0);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(3, "mx_create")
{
    mx_t *a = mx_create(0, 1);
    TEST_IF(a != NULL)
}
TEST_END

TEST_START(4, "mx_create")
{
    mx_t *a = mx_create(1, 1);
    TEST_IF(a == NULL)
    TEST_IF(a->x != 1)
    TEST_IF(a->y != 1)
    mx_destroy(a);
}
TEST_END

TEST_START(5, "mx_mp")
{
    mx_t a = {.x = 2, .y = 3};
    NN_TYPE a_arr[] = { 2.0, 6.0,
                        9.0, 4.0,
                        5.0, 0.0};
    a.arr = a_arr;

    mx_t b = {.x = 3, .y = 2};
    NN_TYPE b_arr[] = { 3.0, 4.0, 7.0,
                        1.0, 9.0, 2.0};
    b.arr = b_arr;

    mx_t out = {.x = 3, .y = 3};
    NN_TYPE out_arr[9];
    out.arr = out_arr;

    mx_mp(a, b, &out, 0);

    NN_TYPE expected_arr[] = {12.0, 62.0, 26.0,
                            31.0, 72.0, 71.0,
                            15.0, 20.0, 35.0};
    for(int i = 0; i < 9; ++i)
    {
        TEST_IF(expected_arr[i] != out.arr[i])
    }
}
TEST_END

TEST_START(6, "mx_mp")
{
    mx_t a = {.x = 2, .y = 3};
    NN_TYPE a_arr[] = { 2.0, 6.0,
                        9.0, 4.0,
                        5.0, 0.0};
    a.arr = a_arr;

    mx_t b = {.x = 2, .y = 3};
    NN_TYPE b_arr[] = { 3.0, 1.0, 
                        4.0, 9.0, 
                        7.0, 2.0};
    b.arr = b_arr;

    mx_t out = {.x = 3, .y = 3};
    NN_TYPE out_arr[9];
    out.arr = out_arr;

    mx_mp(a, b, &out, B);

    NN_TYPE expected_arr[] = {12.0, 62.0, 26.0,
                            31.0, 72.0, 71.0,
                            15.0, 20.0, 35.0};
    for(int i = 0; i < 9; ++i)
    {
        TEST_IF(expected_arr[i] != out.arr[i])
    }
}
TEST_END

int main (void) 
{ 
    int (*test_ptr_arr[])(void) = { test1, test2, test3, test4, test5,
    test6 };
    const int test_size = sizeof(test_ptr_arr)/sizeof(test_ptr_arr[0]);
    int failed = 0;

    for(int i = 0; i < test_size; ++i) failed += (*test_ptr_arr[i])();
    
    printf("%i of %i tests failed\n", failed, test_size);
    return 0; 
}