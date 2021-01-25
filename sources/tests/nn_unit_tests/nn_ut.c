#include "nn.h"
#include "ut_macros.h"

TEST_START(1, "nn_create")
{
    nn_array_t* n = nn_create(100, 15, 2, 
    (nn_params_t){.activ_func = NULL, .drop_rate = 40, .max = 0.2, .min=01, .size = 10},
    (nn_params_t){.activ_func = NULL, .drop_rate = 40, .max = 0.2, .min=01, .size = 20});

    TEST_SIZE(n->layers[0]->delta,  10, 15)
    TEST_SIZE(n->layers[0]->out,    10, 15)
    TEST_SIZE(n->layers[0]->drop,   10, 15)
    TEST_SIZE(n->layers[0]->val,   100, 10)
    TEST_IF(n->layers[0]->drop_rate != 40)

    TEST_SIZE(n->layers[1]->delta,  20, 15)
    TEST_SIZE(n->layers[1]->out,    20, 15)
    TEST_SIZE(n->layers[1]->drop,   20, 15)
    TEST_SIZE(n->layers[1]->val,    10, 20)
    TEST_IF(n->layers[1]->drop_rate != 40)

    TEST_SIZE(n->vdelta, 100,20)

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