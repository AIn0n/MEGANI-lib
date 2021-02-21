#ifndef _NN_H_
#define _NN_H_
#include "mx.h"

//------------------------------------------------MACROS------------------------------------------

#define LAYER_0_NAME DENSE
#define LAYER_0_SETUP dense_setup

#define LAYER_1_NAME DROP
#define LAYER_1_SETUP drop_setup


#define MAX(a, b)	((a) < (b) ? (b) : (a))
#define NO_FUNC ((act_func_t) {.func_cell =NULL, .func_mx =NULL})
#define RELU    ((act_func_t) {.func_cell =relu_deriv_cell, .func_mx =relu_mx})

//------------------------------------------------STRUCTURES---------------------------------------

//TODO DOCS
typedef enum 
{
    LAYER_0_NAME,
    LAYER_1_NAME
} 
layer_type;

typedef struct
{
    NN_TYPE (*func_cell)(NN_TYPE);
    void (*func_mx)(mx_t *);
}
act_func_t;

/** @brief struct used to configure neural layers.
 * 
 *  Structure have cells with every information which we need to build full, neural network
 *  layer. The Structure was built to covers configuration of every layer type. Heavily depend on matrix.h.
 * 
 *  @see nn_create
 */
typedef struct
{
    layer_type  type;
    act_func_t  activ_func;
    uint8_t     drop_rate;
    uint32_t    size;
    NN_TYPE     min;
    NN_TYPE     max;
} 
nn_params_t;

//TODO DOCS
typedef struct 
{
    mx_t*   drop;
    uint8_t drop_rate;
}
drop_data_t;

//TODO DOCS
typedef struct 
{
    mx_t*       val;
    act_func_t  act_func;
} 
dense_data_t;

//TODO DOCS
struct nn_layer_t
{
    mx_t* out;
    mx_t* delta;
    void* data;
    layer_type type;    //TODO: I'm not sure do I need that struct cell.
    void (*forward) (struct nn_layer_t*, const mx_t*);
    void (*backward) (struct nn_layer_t*, mx_t*);
};

/** @brief Main neural network struct.
 * 
 *  This struct is main element of neural network in this library.
 *  It consist of all layers with data, values, outupt, etc (for more info see nn_layer_t),
 *  number of them <size> and additional vdelta matrix used in backpropagation, same for every
 *  layer for memory optimalization.
 * 
 *  @see nn_layer_t
 */
typedef struct 
{
    struct nn_layer_t*  layers; /**< all neurons layers in current network */
    mx_t*               vdelta; /**< vdelta matrix shared between other layers */
    uint16_t            size;   /**< number of layers */
}
nn_array_t;

//------------------------------------------FUNCTIONS--------------------------------------------

/** @brief Create and fill neural network.
 * 
 *  Function check input data, after that it alloc memory for whole structure.
 *  It gives us neural network with <nn_size> layers, every one of them configured
 *  with nn_params_t structures given by <params>.
 * 
 *  On success, pointer to structure is returned. If errors occurs, function return NULL.
 * 
 *  @param [in] in_size input size (width of input matrix)
 *  @param [in] b_size  batch size (heigh of input matrix)
 *  @param [in] nn_size number of layers
 *  @param [in] params  config data for every layer
 */
nn_array_t* 
nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, nn_params_t* params);

/** @brief Free memory allocated for neural network struct.
 * 
 *  Frees memory which we allocated by nn_create() function.
 *  Functions is not thread safe.
 *  
 *  @param [in] nn network which we free
 */
void nn_destroy(nn_array_t *nn);

void nn_predict(nn_array_t* nn, const mx_t* input);

//----------------------------------------ACTIVATION FUNCTIONS--------------------------------------------

void relu_mx(mx_t *a);
NN_TYPE relu_deriv_cell(NN_TYPE a);

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */