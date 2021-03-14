#include "nn.h"
#include "dense.h"
#include "drop.h"

//-----------------------------------USER FUNCTIONS-----------------------------

MX_SIZE (*setup_list[])
(struct nn_layer_t*, MX_SIZE, MX_SIZE, nn_params_t*, setup_params) =
{
    LAYER_0_SETUP,
    LAYER_1_SETUP
};

void
nn_destroy(nn_array_t* nn)
{
    if(nn == NULL) return;
    for(NN_SIZE i = 0; i < nn->size; ++i)
    {
        mx_destroy(nn->layers[i].delta);
        mx_destroy(nn->layers[i].out);
        (*setup_list[nn->layers[i].type])(nn->layers + i, 0, 0, NULL, DELETE);
    }
    free(nn->layers);
    mx_destroy(nn->temp);
    free(nn);
}

nn_array_t* 
nn_create(
    MX_SIZE         input_size, 
    MX_SIZE         b_size,
    uint16_t        nn_size, 
    NN_TYPE         alpha, 
    nn_params_t*    params)
{
    if(!input_size || !b_size || !nn_size) return NULL;
    nn_array_t* ret = (nn_array_t*)calloc(1, sizeof(nn_array_t));
    if(ret == NULL) return NULL;

    ret->layers = (struct nn_layer_t *)calloc
        (nn_size, sizeof(struct nn_layer_t));

    if(ret->layers == NULL) {free(ret); return NULL;}

    MX_SIZE temp_size = 0, layer_in = input_size, err = 0;
    for(NN_SIZE i = 0; i < nn_size; ++i)
    {
        err = (*setup_list[params[i].type])(
            (ret->layers + i), 
            layer_in, 
            b_size, 
            (params + i), 
            CREATE);

        if(!err)
        {
            nn_destroy(ret);
            return NULL;
        }
        temp_size = MAX(temp_size, err);
        layer_in = params[i].size; // layer input size = previous layer output size
    }

    ret->alpha  = alpha;
    ret->size   = nn_size;
    ret->temp   = mx_create(temp_size, 1);
    if(ret->temp == NULL) nn_destroy(ret);

    return ret;
}

void
nn_predict(nn_array_t* nn, const mx_t* input)
{
    const mx_t* prev_out = input;
    for(NN_SIZE i = 0; i < nn->size; ++i)
    {
        nn->layers[i].forward ((nn->layers + i), prev_out);
        prev_out = nn->layers[i].out;
    }
}

void
nn_fit(nn_array_t* nn, const mx_t *input, const mx_t* output)
{
    nn_predict(nn, input);

    const NN_SIZE end = nn->size - 1;
    //delta = output - expected output (last layer case)
    mx_sub(*nn->layers[end].out, *output, nn->layers[end].delta);

    for(NN_SIZE i = end; i > 0; --i)
    {
        nn->layers[i].backward(
            (nn->layers + i), 
            nn, nn->layers[i - 1].out, 
            nn->layers[i - 1].delta);
    }
    //vdelta = delta^T * input
    nn->layers->backward(nn->layers, nn, input, NULL);
}

//---------------------------------ACTIVATION FUNCS-----------------------------
//TODO: split activation funcs to other files or even folder

void relu_mx(mx_t *a)
{
    for(MX_SIZE i = 0; i < a->size; ++i)
        a->arr[i] = MAX(a->arr[i],NN_ZERO);
}

NN_TYPE 
relu_deriv_cell(NN_TYPE a) {return (NN_TYPE)((a > NN_ZERO) ? 1 : 0);}