#include <nn.h>
/*
static nn_layer_t*
nn_layer_create(uint32_t in_size, uint32_t b_size, nn_params_t params)
{
    nn_layer_t* out = (nn_layer_t *)calloc(1, sizeof(nn_layer_t));
    if(out == NULL) return NULL;

    out->activ_func = params.activ_func;
    out->drop_rate = params.drop_rate;

    do {
        out->delta = mx_create(in_size, params.size);
        if(out->delta == NULL) break;

        out->val = mx_create(in_size, params.size);
        if(out->val == NULL) break;

        out->out = mx_create(params.size, b_size);
        if(out->out == NULL) break;

        out->drop = mx_create(in_size, params.size);
        if(out->drop == NULL) break;

        if(params.min && params.max)
        {
            uint32_t val_size = out->val->x * out->val->y;
            for(uint32_t i = 0; i < val_size; ++i)
            {
                //TODO
                out->val->arr[i] = rand();
            }
        }
        return out;
    }while(0);

    mx_destroy(out->delta);
    mx_destroy(out->val);
    mx_destroy(out->out);
    free(out);
    return NULL;
}
*/
nn_array_t* 
nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, ...)
{
    if(!in_size || !b_size || !nn_size) return NULL;

    nn_array_t* out = (nn_array_t *)calloc(1, sizeof(nn_array_t));
    if(out == NULL) return NULL;
    
    out->size = nn_size;
    out->layer = (nn_layer_t *)calloc(nn_size, sizeof(nn_layer_t));
    if(out->layer == NULL)
    {
        free(out);
        return NULL;
    }

    va_list ap;
    va_start(ap, nn_size);

    for(uint16_t i = 0; i < nn_size; ++i)
    {
        //out->layer[i] = nn_layer_create();
    }

    va_end(ap);
    return out;
}
