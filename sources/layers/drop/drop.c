#include "drop.h"

int32_t
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
    if(layer->out == NULL || layer->delta == NULL || !in) return -1;

    drop_data_t* data = (drop_data_t *)calloc(1, sizeof(drop_data_t));
    if(data == NULL) return -1;

    data->drop_rate = params->drop_rate;
    data->drop = mx_create(params->size, batch);
    if(data->drop == NULL) return -1;

    layer->data = (void *)data;
    layer->type = DROP;
    return 0;
}