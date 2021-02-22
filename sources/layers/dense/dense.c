#include "dense.h"

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

int32_t
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
    if(layer->out == NULL || layer->delta == NULL) return -1;

    dense_data_t* data = (dense_data_t *)calloc(1, sizeof(dense_data_t));
    if(data == NULL) return -1;

    data->act_func  = params->activ_func;
    data->val       = mx_create(in, params->size);
    if(data->val == NULL) return -1;

    layer->data     = (void *) data;
    layer->type     = DENSE;
    layer->forward  = (&dense_forward);
    return (int32_t) (in * params->size);
}