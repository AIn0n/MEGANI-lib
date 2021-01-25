#include <nn.h>

static void 
nn_layer_destroy(nn_layer_t* layer)
{
    mx_destroy(layer->delta);
    mx_destroy(layer->val);
    mx_destroy(layer->out);
    mx_destroy(layer->drop);
    free(layer);
}

static nn_layer_t*
nn_layer_create(uint32_t in_size, uint32_t b_size, nn_params_t params)
{
    if(!in_size || !b_size || !(params.size)) return NULL;
    nn_layer_t* out = (nn_layer_t *)calloc(1, sizeof(nn_layer_t));
    if(out == NULL) return NULL;

    out->activ_func = params.activ_func;
    out->drop_rate = params.drop_rate;

    do {
        out->delta  = mx_create(params.size, b_size);
        out->val    = mx_create(in_size, params.size);
        out->out    = mx_create(params.size, b_size);
        out->drop   = mx_create(in_size, params.size);
        
        if(out->delta   == NULL) break;
        if(out->out     == NULL) break;
        if(out->val     == NULL) break;
        if(out->drop    == NULL) break;

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

    nn_layer_destroy(out);
    return NULL;
}

nn_array_t* 
nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, ...)
{
    if(!in_size || !b_size || !nn_size) return NULL;

    nn_array_t* out = (nn_array_t *)calloc(1, sizeof(nn_array_t));
    if(out == NULL) return NULL;
    
    out->size = nn_size;
    out->layers = (nn_layer_t **)calloc(nn_size, sizeof(nn_layer_t *));
    if(out->layers == NULL)
    {
        free(out);
        return NULL;
    }

    va_list ap;
    va_start(ap, nn_size);
    nn_params_t params;
    uint32_t prev_size = 0;
    for(uint16_t i = 0; i < nn_size; ++i)
    {
        params = va_arg(ap, nn_params_t);
        out->layers[i] = nn_layer_create(((i)?(prev_size):(in_size)), b_size, params);
        if(out->layers[i] == NULL)
        {
            out->size = i;
            nn_destroy(out);
            return NULL;
        }
        prev_size = params.size; 
    }
    va_end(ap);
    return out;
}

void nn_destroy(nn_array_t *nn)
{
    if(nn == NULL) return;
    for(uint16_t i = 0; i < nn->size; ++i) nn_layer_destroy(nn->layers[i]);
    free(nn->layers);
    free(nn);
}