#include <nn.h>

//--------------------------------STATIC FUNCTIONS---------------------------------

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
        out->out    = mx_create(params.size, b_size);
        out->val    = mx_create(in_size, params.size);

        if(out->drop_rate > NN_ZERO) 
            out->drop = mx_create(params.size, b_size);

        if(out->delta   == NULL) break;
        if(out->out     == NULL) break;
        if(out->val     == NULL) break;
        if((out->drop_rate > NN_ZERO) && (out->drop == NULL)) break;

        if(params.min && params.max)
        {
            uint32_t val_size = out->val->x * out->val->y;
            NN_TYPE a;
            NN_TYPE diff = (params.max - params.min);
            for(uint32_t i = 0; i < val_size; ++i)
            {
                a = (NN_TYPE) rand() / RAND_MAX;
                out->val->arr[i] = params.min + a * diff;
            }
        }
        return out;
    }while(0);

    nn_layer_destroy(out);
    return NULL;
}

static void 
nn_layer_drop_reroll(nn_layer_t* layer)
{
    uint32_t size = layer->drop->x * layer->drop->y;
    for(uint32_t i = 0; i < size; ++i)
    {
        layer->drop->arr[i] = ((rand() % 100) >= layer->drop_rate);
    }
}

//-----------------------------------USER FUNCTIONS---------------------------------

nn_array_t* 
nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, ...)
{
    if(!in_size || !b_size || !nn_size) return NULL;

    nn_array_t* out = (nn_array_t *)calloc(1, sizeof(nn_array_t));
    if(out == NULL) return NULL;
    
    out->size = nn_size;
    out->layers = (nn_layer_t **)calloc(nn_size, sizeof(nn_layer_t *));
    if(out->layers == NULL) { free(out); return NULL; }

    va_list ap;
    va_start(ap, nn_size);

    nn_params_t params;
    uint32_t    size = in_size;             //size(layer->output) == size(next->layer->input)
    uint32_t    max_in = 0, max_out = 0;    //this vars we use to build vdelta matrix

    for(uint16_t i = 0; i < nn_size; ++i)
    {
        params = va_arg(ap, nn_params_t);
        out->layers[i] = nn_layer_create(size, b_size, params);
        if(out->layers[i] == NULL)
        {
            out->size = i;
            nn_destroy(out);
            return NULL;
        }
        max_in = MAX(size, max_in);
        max_out= MAX(params.size, max_out);
        size = params.size; 
    }
    va_end(ap);

    out->vdelta = mx_create(max_in, max_out);
    if(out->vdelta == NULL) nn_destroy(out);

    return out;
}

void 
nn_destroy(nn_array_t *nn)
{
    if(nn == NULL) return;
    for(uint16_t i = 0; i < nn->size; ++i) nn_layer_destroy(nn->layers[i]);
    mx_destroy(nn->vdelta);
    free(nn->layers);
    free(nn);
    nn = NULL;
}

void 
nn_predict(nn_array_t *nn,const mx_t* input, uint8_t flags)
{
    const mx_t* prev_out = input;
    nn_layer_t** l = nn->layers;
    for(uint16_t i = 0; i < nn->size; ++i)
    {
    //layer output = layer input * layer values T
        mx_mp(*prev_out, *(l[i]->val),  l[i]->out, B);
        prev_out = l[i]->out;

    //layer output = activation function ( layer output )
        if(l[i]->activ_func.func_mx != NULL)
            (*l[i]->activ_func.func_mx)(l[i]->out);

    //layer output = dropout mask ( layer output )
        if((flags & 1) && l[i]->drop_rate)
        {
            nn_layer_drop_reroll(nn->layers[i]);
            mx_hadamard(*(l[i]->out), *(l[i]->drop), l[i]->out);
            mx_mp_num(l[i]->out, (l[i]->drop_rate / 100));
        }
    }
}

//TODO
void 
nn_fit(nn_array_t *nn, const mx_t* in, const mx_t* out, NN_TYPE alpha)
{
//counting output
    nn_predict(nn, in, 0);

    nn_layer_t** l = nn->layers;
    int32_t n_size = (nn->size - 1);

//counting delta    
    for(int32_t i = n_size; i > -1; --i)
    {
    //delta = out - expected out    
    //delta = next delta * next values (last layer case)
        if(i == n_size) 
            mx_sub(*l[i]->out, *out, l[i]->delta);
        else            
            mx_mp(*(l[i+1]->delta), *(l[i+1]->val), l[i]->delta, DEF);

    //delta = delta o activation function ( output )
        if(l[i]->activ_func.func_cell != NULL)
            mx_hadam_lambda(l[i]->delta, *l[i]->out, l[i]->activ_func.func_cell);
    //delta = delta o dropout mask
        if(l[i]->drop_rate > NN_ZERO)
            mx_hadamard(*l[i]->delta, *l[i]->drop, l[i]->delta);
    }

//values update
    const mx_t *prev_out = in;
    for(uint32_t i = 0; i < nn->size; ++i)
    {
    //vdelta is universal for every matrix so before use we have to resize it
        nn->vdelta->x = l[i]->val->x;
        nn->vdelta->y = l[i]->val->y;
    //value delta = delta^T * previous output
    //value delta = delat^T * input (first layer case)
        mx_mp(*l[i]->delta, *prev_out, nn->vdelta, A);
    //value delta = value delta * alpha
        mx_mp_num(nn->vdelta, alpha);
    //value = value - value delta
        mx_sub(*l[i]->val, *nn->vdelta, l[i]->val);

        prev_out = l[i]->out;
    }
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