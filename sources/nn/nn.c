#include <nn.h>

nn_array* 
nn_create(void)
{
    nn_array* out = (nn_array *)calloc(1, sizeof(nn_layer));
    if(out != NULL) out[0] = NULL;
    return out;
}

//NOT DONE YET!
uint8_t 
nn_add_layer(nn_array* nn, uint32_t neurons, uint32_t in_size, 
void (*activ_func)(NN_TYPE*, uint8_t), uint32_t batch_size)
{
    if(nn == NULL) return 1;
}