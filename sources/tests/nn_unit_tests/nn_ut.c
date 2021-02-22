#include "nn.h"
#include "macros_ut.h"
#include "dense.h"
#include "drop.h"

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

    dense_data_t* ptr1 = n->layers[0].data;
    TEST_SIZE(ptr1->val,     100, 10)

    TEST_SIZE(n->layers[1].delta,  10, 15)
    TEST_SIZE(n->layers[1].out,    10, 15)
    
    drop_data_t* ptr2 = n->layers[1].data;
    TEST_SIZE(ptr2->mask, 10, 15)
    TEST_IF(ptr2->drop_rate != 40)
    TEST_IF(n->layers[1].type != DROP)
    
    TEST_SIZE(n->layers[2].delta,   20, 15)
    TEST_SIZE(n->layers[2].out,     20, 15)
    TEST_IF(n->layers[2].type != DENSE)

    ptr1 =(dense_data_t *) n->layers[2].data;
    TEST_SIZE(ptr1->val,    10, 20)

    TEST_SIZE(n->vdelta, 100,10)

    nn_destroy(n);
}
TEST_END

TEST_START(2, "nn_predict")
{
    nn_params_t initializer[] = 
    {
        {.type = DENSE, .activ_func = NO_FUNC, .size = 3},
        {.type = DENSE, .activ_func = NO_FUNC, .size = 3}
    };

    NN_TYPE input[3] = {8.5, 0.65, 1.2};
    mx_t mx_input = {.arr = input, .x = 3, .y = 1};

    nn_array_t* n = nn_create(3, 1, 2, initializer);

    dense_data_t* dense_ptr = (n->layers->data);
    NN_TYPE val0[9] = { 0.1,    0.2,   -0.1,
                       -0.1,    0.1,    0.9,
                        0.1,    0.4,    0.1};
    MX_CPY(val0, dense_ptr->val)

    NN_TYPE val1[9] = { 0.3,    1.1,   -0.3,
                        0.1,    0.2,    0.0,
                        0.0,    1.3,    0.1};
    dense_ptr = n->layers[1].data;
    MX_CPY(val1, dense_ptr->val)

    nn_predict(n, &mx_input);

    NN_TYPE expected_output[3] = {0.2135, 0.145, 0.5065};
    for(uint32_t i = 0; i < 3; ++i)
    {
        TEST_ERR(n->layers[1].out->arr[i], expected_output[i], 0.0001)
    }

    nn_destroy(n);
}
TEST_END

int nn_ut(void) 
{
    int (*test_ptr_arr[])(void) = { test1, test2};
    const int test_size = sizeof(test_ptr_arr)/sizeof(test_ptr_arr[0]);
    int failed = 0;

    puts("\nneural networks unit tests\n");

    for(int i = 0; i < test_size; ++i) failed += (*test_ptr_arr[i])();
    
    printf("%i of %i tests failed\n", failed, test_size);
    return failed; 
}