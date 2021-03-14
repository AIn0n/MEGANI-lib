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

/** @def NN_SIZE
 *  @brief Type for number of layer in neural network. By default it is uint16_t.
 */
#define NN_SIZE uint16_t

//------------------------------------------------STRUCTURES---------------------------------------

/** @brief enum with all types of layers which we can use in our networks.
 * 
 *  In this typedef we have predefined before names of layers which we can use
 *  in neural networks. I made this structure to make whole library as much both
 *  extenisble and shrinkable as possible. If you want to insert some other, non default layers
 *  here, please remember also about layer_n_setup macros!
 * 
 */
typedef enum 
{
    LAYER_0_NAME = 0,
    LAYER_1_NAME = 1
} 
layer_type;


/** @brief Simple enum with function purpose.
 * 
 *  I created this enum to make functions which destroy and create new layer to make it easier.
 *  It is used internally and you don't have to play with it by yourself.
 */
typedef enum {
    CREATE,
    DELETE
}setup_params;

/** @brief Structure with all activations functions subroutines used in single layer.
 * 
 *  Structure consist of two function pointers - one of them is used in forwarding
 *  on whole matrix, second one is used in backpropagation as function derivative 
 *  with single matrix cell (function mx_hadam_lambda). I do this in that way to 
 *  do as small amount of operations on matrices as possible.
 * 
 *  @see mx_hadam_lambda
 */
typedef struct
{
    MX_TYPE (*func_cell)(MX_TYPE);  /**< function used in forwarding */
    void (*func_mx)(mx_t *);        /**< function used in backpropagation */
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
    MX_SIZE    size;
    MX_TYPE     min;
    MX_TYPE     max;
} 
nn_params_t;

/** @brief Main neural network struct.
 * 
 *  This struct is main element of neural network in this library.
 *  It consist of all layers with data, values, outupt, etc (for more info see nn_layer_t),
 *  number of them <size> and additional temporary matrix used in backpropagation and cnn, same for every
 *  layer for memory optimalization.
 * 
 *  @see nn_layer_t
 */
typedef struct 
{
    struct nn_layer_t*  layers; /**< all neurons layers in current network */
    mx_t*               temp; /**< temporary matrix shared between layers for things like im2col, value delta, etc */
    MX_TYPE             alpha;  /**< alpha indicates learning speed */
    NN_SIZE             size;   /**< number of layers */
}
nn_array_t;

/** @brief struct containing all data from single layer.
 * 
 *  This structure have every learnable and constant data from single layer.
 *  Cells like out, delta, data are universal for every layer, more focused structures,
 *  functions etc, based on layer purpose you can find in data field which is pointer
 *  to other structure (example: dense_data_t). Functions forward and backward are used
 *  in backpropagation.
 */
struct nn_layer_t
{
    mx_t* out;          /**< layer output */
    mx_t* delta;        /**< layer delta */
    void* data;         /**< layer specialized data, like activation functions in dense */
    layer_type type;    /**< layer type (used in nn_destroy()) */
    void (*forward) (struct nn_layer_t*, const mx_t*);                      /**< function used in nn_predict() */
    void (*backward) (struct nn_layer_t*, nn_array_t*, const mx_t*, mx_t*); /**< function use in nn_fit() */
};

//------------------------------------------FUNCTIONS--------------------------------------------

/** @brief Create and fill neural network.
 * 
 *  Function check input data, after that it alloc memory for whole structure.
 *  It gives us neural network with <nn_size> layers, every one of them configured
 *  with nn_params_t structures given by <params>.
 * 
 *  On success, pointer to structure is returned. If errors occurs, function return NULL.
 * 
 *  @param [in] input_size input size (width of input matrix)
 *  @param [in] alpha   alpha learning rate
 *  @param [in] b_size  batch size (heigh of input matrix)
 *  @param [in] nn_size number of layers
 *  @param [in] params  config data for every layer
 */
nn_array_t* 
nn_create(MX_SIZE input_size, MX_SIZE b_size, NN_SIZE nn_size, MX_TYPE alpha, nn_params_t* params);

/** @brief Free memory allocated for neural network struct.
 * 
 *  Frees memory which we allocated by nn_create() function.
 *  Functions is not thread safe.
 *  
 *  @param [in] nn network which we free
 */
void nn_destroy(nn_array_t *nn);

//TODO docs
void nn_predict(nn_array_t* nn, const mx_t* input);
void nn_fit(nn_array_t* nn, const mx_t *input, const mx_t* output);

//----------------------------------------ACTIVATION FUNCTIONS--------------------------------------------
//TODO: Im not sure, but maybe activations funcs will get new file only for them

void relu_mx(mx_t *a);
MX_TYPE relu_deriv_cell(MX_TYPE a);

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */