#include <nn.h>

//--------------------------------STATIC FUNCTIONS---------------------------------
//TODO: all of this func have to get new file

typedef enum {
    CREATE,
    DELETE
}setup_params;

void
dense_forward(struct nn_layer_t* self, const mx_t * input)
{
    //output = input * values ^T
    dense_data_t* data = self->data;
    mx_mp(*input, *data->val, self->out, B);  

    //layer output = activation function ( layer output )
    if(*data->act_func.func_mx != NULL)
        (*data->act_func.func_mx)(self->out);  
}

static int32_t
dense_setup(struct nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t* params, setup_params purpose)
{
    if(purpose == DELETE)
    {
        dense_data_t* data = (dense_data_t *)layer->data;
        if(data != NULL)
        {
            mx_destroy(data->val);
            free(data);
        }
        return 0;
    }
    layer->out    = mx_create(params->size, batch);
    layer->delta  = mx_create(params->size, batch);
    if(layer->out == NULL || layer->delta == NULL) return 1;

    dense_data_t* data = (dense_data_t *)calloc(1, sizeof(dense_data_t));
    if(data == NULL) return 1;

    data->val   = mx_create(in, params->size);
    if(data->val == NULL) return 1;

    data->act_func = params->activ_func;
    layer->data = (void *) data;
    layer->type = DENSE;
    layer->forward = (&dense_forward);
    return (int32_t) (in * params->size);
}

static int32_t
drop_setup(struct nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t* params, setup_params purpose)
{
    if(purpose == DELETE)
    {
        drop_data_t* data = (drop_data_t *)layer->data;
        if(data != NULL)
        {
            mx_destroy(data->drop);
            free(data);
        }
        return 0;
    }
    params->size = (params - 1)->size;
    layer->out = mx_create(params->size, batch);
    layer->delta = mx_create(params->size, batch);
    if(layer->out == NULL || layer->delta == NULL || !in) return 1;

    drop_data_t* data = (drop_data_t *)calloc(1, sizeof(drop_data_t));
    if(data == NULL) return 1;

    data->drop = mx_create(params->size, batch);
    if(data->drop == NULL) return 1;

    data->drop_rate = params->drop_rate;
    layer->data = (void *)data;
    layer->type = DROP;
    return 0;
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
            case LAYER_0_NAME:
                LAYER_0_SETUP(nn->layers + i, 0, 0, NULL, DELETE);
                break;

            case LAYER_1_NAME:
                LAYER_1_SETUP(nn->layers + i, 0, 0, NULL, DELETE);
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

    ret->layers = (struct nn_layer_t *)calloc(nn_size, sizeof(struct nn_layer_t));
    if(ret->layers == NULL) {free(ret); return NULL;}

    uint32_t max_delta_size = -1, max_delta_x = 0, max_delta_y = 0;
    int32_t err = -1;
    uint32_t layer_in = in_size;    
    for(uint16_t i = 0; i < nn_size; ++i)
    {
        switch(params[i].type)
        {
            case LAYER_0_NAME:
                err = LAYER_0_SETUP((ret->layers + i), layer_in, b_size, (params + i), CREATE);
                break;
            case LAYER_1_NAME:
                err = LAYER_1_SETUP((ret->layers + i), layer_in, b_size, (params + i), CREATE);
                break;
        }
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