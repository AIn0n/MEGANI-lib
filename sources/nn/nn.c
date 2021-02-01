#include <nn.h>

//--------------------------------STATIC FUNCTIONS---------------------------------

static uint8_t
neural_create(nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t params)
{
    layer->out    = mx_create(params.size, batch);
    layer->delta  = mx_create(params.size, batch);
    if(layer->out == NULL || layer->delta == NULL) return 1;

    neural_data_t* data = (neural_data_t *)calloc(1, sizeof(neural_data_t));
    if(data == NULL) return 1;

    data->val   = mx_create(in, params.size);
    if(data->val == NULL) return 1;

    data->act_func = params.activ_func;
    layer->data = (void *) data;
    layer->type = NEURAL;
    return 0;
}

static void
neural_destroy(nn_layer_t* layer)
{
    neural_data_t* data = (neural_data_t *)layer->data;
    if(data != NULL)
    {
        mx_destroy(data->val);
        free(data);
    }
}

static uint8_t
drop_create(nn_layer_t* layer, uint32_t batch, nn_params_t params)
{
    layer->out = mx_create(params.size, batch);
    layer->delta = mx_create(params.size, batch);
    if(layer->out == NULL || layer->delta == NULL) return 1;

    drop_data_t* data = (drop_data_t *)calloc(1, sizeof(drop_data_t));
    if(data == NULL) return 1;

    data->drop = mx_create(params.size, batch);
    if(data->drop == NULL) return 1;

    data->drop_rate = params.drop_rate;
    layer->data = (void *)data;
    layer->type = DROP;
    return 0;
}

static void
drop_destroy(nn_layer_t *layer)
{
    drop_data_t* data = (drop_data_t *)layer->data;
    if(data != NULL)
    {
        mx_destroy(data->drop);
        free(data);
    }
}

//-----------------------------------USER FUNCTIONS---------------------------------

void
nn_destroy(nn_array_t* nn)
{
    if(nn == NULL) return;
    for(uint16_t i = 0; i < nn->size; ++i)
    {
        mx_destroy(nn->layers[i].delta);
        mx_destroy(nn->layers[i].out);
        switch(nn->layers[i].type)
        {
            case NEURAL:
                neural_destroy(nn->layers + i);
                break;

            case DROP:
                drop_destroy(nn->layers + i);
                break;
        }
    }
    free(nn->layers);
    mx_destroy(nn->vdelta);
    free(nn);
}

nn_array_t* 
nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, nn_params_t* params)
{
    if(!in_size || !b_size || !nn_size) return NULL;
    nn_array_t* ret = (nn_array_t*)calloc(1, sizeof(nn_array_t));
    if(ret == NULL) return NULL;

    ret->layers = (nn_layer_t *)calloc(nn_size, sizeof(nn_layer_t));
    if(ret->layers == NULL) {free(ret); return NULL;}

    uint8_t err;
    uint32_t layer_in = in_size;    
    uint32_t max_in = 0, max_out = 0, max_size = 0; //used to build shared vdelta matrix
    for(uint16_t i = 0; i < nn_size; ++i)
    {
        switch(params[i].type)
        {
            case NEURAL:
                err = neural_create((ret->layers + i), layer_in, b_size, params[i]);
                if((layer_in * params[i].size) > max_size)
                {    
                    max_in = layer_in;
                    max_out = params[i].size;
                    max_size = max_in * max_out;
                }
                break;
            case DROP:
                if(i) params[i].size = params[i-1].size;    //size setup for drop layer
                err = drop_create((ret->layers + i), b_size, params[i]);
                break;
        }
        if(err)
        {
            nn_destroy(ret);
            return NULL;
        }
        layer_in = params[i].size; // layer input size = previous layer output size
    }

    ret->size = nn_size;
    ret->vdelta = mx_create(max_in, max_out);
    if(ret->vdelta == NULL) nn_destroy(ret);

    return ret;
}

//---------------------------------ACTIVATION FUNCS------------------------------------------

void relu_mx(mx_t *a)
{
    uint32_t size = a->x * a->y;
    for(uint32_t i = 0; i < size; ++i)
        a->arr[i] = MAX(a->arr[i],NN_ZERO);
}

NN_TYPE 
relu_deriv_cell(NN_TYPE a) {return (NN_TYPE)((a > NN_ZERO) ? 1 : 0);}