#include "drop.h"

//--------------------------------------STATIC FUNCTIONS-----------------------------------------------------

void
drop_reroll(drop_data_t *data)
{
    uint32_t size = data->mask->x * data->mask->y;
    for(uint32_t i = 0; i < size; ++i)
    {
        data->mask->arr[i] = ((rand() % 100) >= data->drop_rate);
    }
}

//--------------------------------------PUBLIC FUNCTIONS------------------------------------------------------

void
drop_forward(struct nn_layer_t* self, const mx_t * input)
{
    drop_data_t* data = (drop_data_t*) self->data;
    drop_reroll(data);  //randomize dropout mask values
    mx_hadamard(*input, *data->mask, self->out);    //output = dropout mask ( output )
    mx_mp_num(self->out, (data->drop_rate/100));
    
}

void
drop_backward(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta)
{
    if(n == NULL || prev_out == NULL) return;
    drop_data_t* data = (drop_data_t *) self->data;
    mx_hadamard(*self->delta, *data->mask, prev_delta);
}

int32_t
drop_setup(struct nn_layer_t* layer, uint32_t in, uint32_t batch, nn_params_t* params, setup_params purpose)
{
    if(purpose == DELETE)
    {
        drop_data_t* data = (drop_data_t *)layer->data;
        if(data != NULL)
        {
            mx_destroy(data->mask);
            free(data);
        }
        return 0;
    }
    params->size = (params - 1)->size;
    layer->out = mx_create(params->size, batch);
    layer->delta = mx_create(params->size, batch);
    if(layer->out == NULL || layer->delta == NULL || !in) return -1;

    drop_data_t* data = (drop_data_t *)calloc(1, sizeof(drop_data_t));
    if(data == NULL) return -1;

    data->drop_rate = params->drop_rate;
    data->mask = mx_create(params->size, batch);
    if(data->mask == NULL) return -1;

    layer->data = (void *)data;
    layer->type = DROP;
    layer->forward = (&drop_forward);
    layer->backward = (&drop_backward);
    return 0;
}