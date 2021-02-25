#include "dense.h"

//----------------------STATIC FUNCTIONS---------------------------------------------------

static void
dense_fill_rng(mx_t* values, nn_params_t* params)
{
    NN_TYPE rand_val, diff = (params->max - params->min);
    for(MX_SIZE i = 0; i < values->size; ++i)
    {
        rand_val = (NN_TYPE) rand() / RAND_MAX;
        values->arr[i] = params->min + rand_val * diff;
    }
}

//----------------------------PUBLIC FUNCTIONS----------------------------------------------

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

void 
dense_backward(struct nn_layer_t* self, nn_array_t* n, const mx_t* prev_out, mx_t* prev_delta)
{
    dense_data_t* data = (dense_data_t *) self->data;    
    if(data->act_func.func_cell != NULL) //delta = delta o activation function ( output )
    {
        mx_hadam_lambda(self->delta, *self->out, data->act_func.func_cell);
    }
    n->vdelta->x = data->val->x;   //vdelta is shared between layers so we had to change the size
    n->vdelta->y = data->val->y;

    if(prev_delta != NULL)                          //prev delta = curr delta * curr values
        mx_mp(*self->delta, *data->val, prev_delta, DEF);

    mx_mp(*self->delta, *prev_out, n->vdelta, A);   //value delta = delta^T * previous output
    mx_mp_num(n->vdelta, n->alpha);                 //value delta = value delta * alpha
    mx_sub(*data->val, *n->vdelta, data->val);      //values = values - vdelta
}

MX_SIZE
dense_setup(struct nn_layer_t* self, MX_SIZE in, MX_SIZE batch, nn_params_t* params, setup_params purpose)
{
    if(purpose == DELETE)
    {
        dense_data_t* data = (dense_data_t *)self->data;
        if(data != NULL)
        {
            mx_destroy(data->val);
            free(data);
        }
        return 0;
    }
    self->out    = mx_create(params->size, batch);
    self->delta  = mx_create(params->size, batch);
    if(self->out == NULL || self->delta == NULL) return 0;

    dense_data_t* data = (dense_data_t *)calloc(1, sizeof(dense_data_t));
    if(data == NULL) return 0;

    data->act_func  = params->activ_func;
    data->val       = mx_create(in, params->size);
    if(data->val == NULL) return 0;
    if(params->min && params->max) dense_fill_rng(data->val, params);

    self->data     = (void *) data;
    self->type     = DENSE;
    self->forward  = (&dense_forward);
    self->backward = (&dense_backward);
    return (in * params->size);
}