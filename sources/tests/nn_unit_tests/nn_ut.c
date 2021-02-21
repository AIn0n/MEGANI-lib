#include "nn.h"
#include "ut_macros.h"

TEST_START(1, "nn_create")
{
    nn_params_t initializer[] = 
    {
        {.type = DENSE, .activ_func = RELU, .max = 0.2, .min=0.1, .size = 10},
        {.type = DROP, .drop_rate = 40},
        {.type = DENSE, .activ_func = NO_FUNC, .max = 0.2, .min=01, .size = 20}
    };
    nn_array_t* n = nn_create(100, 15, 3, initializer);

    TEST_SIZE(n->layers[0].delta,   10, 15)
    TEST_SIZE(n->layers[0].out,     10, 15)
    TEST_IF(n->layers[0].type != DENSE)

    neural_data_t* ptr1 = n->layers[0].data;
    TEST_SIZE(ptr1->val,     100, 10)

    TEST_SIZE(n->layers[1].delta,  10, 15)
    TEST_SIZE(n->layers[1].out,    10, 15)
    
    drop_data_t* ptr2 = n->layers[1].data;
    TEST_SIZE(ptr2->drop, 10, 15)
    TEST_IF(ptr2->drop_rate != 40)
    TEST_IF(n->layers[1].type != DROP)
    
    TEST_SIZE(n->layers[2].delta,   20, 15)
    TEST_SIZE(n->layers[2].out,     20, 15)
    TEST_IF(n->layers[2].type != DENSE)

    ptr1 =(neural_data_t *) n->layers[2].data;
    TEST_SIZE(ptr1->val,    10, 20)

    TEST_SIZE(n->vdelta, 100,10)

    nn_destroy(n);
}
TEST_END

int nn_ut(void) 
{
    int (*test_ptr_arr[])(void) = { test1};
    const int test_size = sizeof(test_ptr_arr)/sizeof(test_ptr_arr[0]);
    int failed = 0;

    puts("\nneural networks unit tests\n");

    for(int i = 0; i < test_size; ++i) failed += (*test_ptr_arr[i])();
    
    printf("%i of %i tests failed\n", failed, test_size);
    return failed; 
}