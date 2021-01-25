#include "nn.h"
#include "ut_macros.h"

int nn_ut(void) 
{
    nn_params_t params1 = {.activ_func = NULL, .drop_rate = 0.4, .max = 0.2, .min=01, .size = 10};
    nn_params_t params2 = {.activ_func = NULL, .drop_rate = 0.4, .max = 0.2, .min=01, .size = 20};
    nn_array_t* n = nn_create(100, 15, 2, params1, params2);
    nn_destroy(n);
    if(n) return 1; 
    return 0;
}