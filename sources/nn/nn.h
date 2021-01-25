#ifndef _NN_H_
#define _NN_H_
#include "mx.h"
#include <stdarg.h>

#define MAX(a, b)	((a) < (b) ? (b) : (a))

/** @struct nn_layer
 *  @brief Structure with single neural network layer.
 * 
 *  I designed this struct as small and fast as possible.
 *  It depend matrix.h file for structs and functions for
 *  matrix.
 * 
 *  @see mx_t 
 */
typedef struct 
{
    mx_t *out;           /**< matrix with layer output */
    mx_t *val;           /**< values of neurons in current layer */
    mx_t *delta;         /**< delta, used in backpropagation */
    mx_t *drop;          /**< dropout mask */
    double drop_rate;  /**< percentage amount of turned off neurons in dropout */
    void (*activ_func)(NN_TYPE*, uint8_t);  /**< activation function pointer */
} 
nn_layer_t;

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
    nn_layer_t**    layers; /**< all neurons layers in this network */
    mx_t*           vdelta; /**< vdelta matrix shared between other layers */
    uint16_t        size;   /**< number of layers */
}
nn_array_t;

typedef struct 
{
    uint32_t    size;
    void        (*activ_func)(NN_TYPE*, uint8_t);
    double      drop_rate;
    NN_TYPE     min;
    NN_TYPE     max;
} 
nn_params_t;

/** @brief Create and fill neural network.
 * 
 *  Function check input data, after that it alloc memory for whole structure.
 *  It gives us neural network with <nn_size> layers, every one of them configured
 *  with nn_params_t structures given in place of <...>.
 * 
 *  On success, pointer to structure is returned. If errors occurs, function return NULL.
 * 
 *  @param [in] in_size input size (width of input matrix)
 *  @param [in] b_size  batch size (heigh of input matrix)
 *  @param [in] nn_size numer of layers
 */
nn_array_t* nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, ...);

void nn_destroy(nn_array_t *nn);

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */