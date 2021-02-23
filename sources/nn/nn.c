#include "nn.h"
#include "dense.h"
#include "drop.h"

//-----------------------------------USER FUNCTIONS---------------------------------

int32_t (*setup_list[])(struct nn_layer_t*, uint32_t, uint32_t, nn_params_t*, setup_params) =
{
    LAYER_0_SETUP,
    LAYER_1_SETUP
};

void
nn_destroy(nn_array_t* nn)
{
    if(nn == NULL) return;
    for(uint16_t i = 0; i < nn->size; ++i)
    {
        mx_destroy(nn->layers[i].delta);
        mx_destroy(nn->layers[i].out);
        (*setup_list[nn->layers[i].type])(nn->layers + i, 0, 0, NULL, DELETE);
    }
    free(nn->layers);
    mx_destroy(nn->vdelta);
    free(nn);
}

nn_array_t* 
nn_create(uint32_t input_size, uint32_t b_size, uint16_t nn_size, nn_params_t* params)
{
    if(!input_size || !b_size || !nn_size) return NULL;
    nn_array_t* ret = (nn_array_t*)calloc(1, sizeof(nn_array_t));
    if(ret == NULL) return NULL;

    ret->layers = (struct nn_layer_t *)calloc(nn_size, sizeof(struct nn_layer_t));
    if(ret->layers == NULL) {free(ret); return NULL;}

    uint32_t max_delta_size = -1, max_delta_x = 0, max_delta_y = 0, layer_in = input_size;
    int32_t err = -1;
    for(uint16_t i = 0; i < nn_size; ++i)
    {
        err = (*setup_list[params[i].type])((ret->layers + i), layer_in, b_size, (params + i), CREATE);
        if(err == -1)
        {
            nn_destroy(ret);
            return NULL;
        }
        if((int32_t) max_delta_size < err)
        {
            max_delta_size = err;
            max_delta_x = layer_in;
            max_delta_y = params[i].size;
        }
        layer_in = params[i].size; // layer input size = previous layer output size
    }

    ret->size = nn_size;
    ret->vdelta = mx_create(max_delta_x, max_delta_y);
    if(ret->vdelta == NULL) nn_destroy(ret);

    return ret;
}

void
nn_predict(nn_array_t* nn, const mx_t* input)
{
    const mx_t* prev_out = input;
    for(uint16_t i = 0; i < nn->size; ++i)
    {
        nn->layers[i].forward ((nn->layers + i), prev_out);
        prev_out = nn->layers[i].out;
    }
}

void
nn_fit(nn_array_t* nn, const mx_t *input, const mx_t* output, NN_TYPE alpha)
{
    nn_predict(nn, input);
    nn->alpha = alpha;

    //delta = output - expected output (last layer case)
    mx_sub(*nn->layers[nn->size - 1].out, *output, nn->layers[nn->size - 1].delta);
    
    for(uint16_t i = nn->size - 1; i > 0; --i)
    {
        nn->layers[i].backward((nn->layers + i), nn, nn->layers[i - 1].out, nn->layers[i - 1].delta);
    }
    //vdelta = delta^T * input
    nn->layers->backward(nn->layers, nn, input, NULL);
}

//---------------------------------ACTIVATION FUNCS------------------------------------------
//TODO: split activation funcs to other files or even folder

void relu_mx(mx_t *a)
{
    uint32_t size = a->x * a->y;
    for(uint32_t i = 0; i < size; ++i)
        a->arr[i] = MAX(a->arr[i],NN_ZERO);
}

NN_TYPE 
relu_deriv_cell(NN_TYPE a) {return (NN_TYPE)((a > NN_ZERO) ? 1 : 0);}