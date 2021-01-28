#include "nn.h"
#include "ut_macros.h"

TEST_START(1, "nn_create")
{
    nn_array_t* n = nn_create(100, 15, 2, 
    (nn_params_t){.activ_func = RELU, .drop_rate = 40, .max = 0.2, .min=01, .size = 10},
    (nn_params_t){.activ_func = NO_FUNC, .drop_rate = 40, .max = 0.2, .min=01, .size = 20});

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

TEST_START(2, "nn_predict")
{
    nn_array_t nn;
    
    NN_TYPE val0[9] = { 0.1,    0.2,   -0.1,
                       -0.1,    0.1,    0.9,
                        0.1,    0.4,    0.1};
    
    NN_TYPE val1[9] = { 0.3,    1.1,   -0.3,
                        0.1,    0.2,    0.0,
                        0.0,    1.3,    0.1};

    NN_TYPE input[3] = {8.5, 0.65, 1.2};

    mx_t mx_input ={.arr = input,.x = 3, .y = 1};
    mx_t mx_val0 = {.arr = val0, .x = 3, .y = 3};
    mx_t mx_val1 = {.arr = val1, .x = 3, .y = 3};
    
    NN_TYPE out0[3];
    NN_TYPE out1[3];

    mx_t mx_out0 = {.arr = out0, .x = 3, .y = 1};
    mx_t mx_out1 = {.arr = out1, .x = 3, .y = 1};

    nn_layer_t layer0;
    layer0.activ_func = NO_FUNC;
    layer0.out = &mx_out0;
    layer0.val = &mx_val0;

    nn_layer_t layer1;
    layer1.activ_func = NO_FUNC;
    layer1.out = &mx_out1;
    layer1.val = &mx_val1;

    nn_layer_t* layers[2] = { &layer0, &layer1};
    nn.layers = layers;
    nn.size = 2;

    nn_predict(&nn, &mx_input, 0);

    NN_TYPE expected_output[3] = {0.2135, 0.145, 0.5065};
    for(uint32_t i = 0; i < 3; ++i)
    {
        TEST_ERR(nn.layers[1]->out->arr[i], expected_output[i], 0.0001)
    }
}
TEST_END

TEST_START(3, "nn_predict")
{
    nn_array_t nn;
    
    NN_TYPE val0[3] = { 0.1, -0.1, 0.1};
    NN_TYPE val1[3] = { 0.3, 1.1, -0.3};

    NN_TYPE input[1] = {8.5};

    mx_t mx_input ={.arr = input,.x = 1, .y = 1};
    mx_t mx_val0 = {.arr = val0, .x = 1, .y = 3};
    mx_t mx_val1 = {.arr = val1, .x = 3, .y = 1};
    
    NN_TYPE out0[3];
    NN_TYPE out1[1];

    mx_t mx_out0 = {.arr = out0, .x = 3, .y = 1};
    mx_t mx_out1 = {.arr = out1, .x = 1, .y = 1};

    nn_layer_t layer0;
    layer0.activ_func = RELU;
    layer0.out = &mx_out0;
    layer0.val = &mx_val0;

    nn_layer_t layer1;
    layer1.activ_func = NO_FUNC;
    layer1.out = &mx_out1;
    layer1.val = &mx_val1;

    nn_layer_t* layers[2] = { &layer0, &layer1};
    nn.layers = layers;
    nn.size = 2;

    nn_predict(&nn, &mx_input, 0);

    TEST_ERR(nn.layers[1]->out->arr[0], 0.0, 0.0001)
}
TEST_END

int nn_ut(void) 
{
    int (*test_ptr_arr[])(void) = { test1, test2, test3};
    const int test_size = sizeof(test_ptr_arr)/sizeof(test_ptr_arr[0]);
    int failed = 0;

    puts("\nneural networks unit tests\n");

    for(int i = 0; i < test_size; ++i) failed += (*test_ptr_arr[i])();
    
    printf("%i of %i tests failed\n", failed, test_size);
    return failed; 
}