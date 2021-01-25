#ifndef _NN_H_
#define _NN_H
#include "mx.h"
#include <stdarg.h>

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

typedef struct 
{
    nn_layer_t** layers;
    uint16_t    size; 
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
 *  It gives us neural network with <nol> layers, every one of them configured
 *  with nn_params_t structures given in place of <...>.
 * 
 *  On success, pointer to structure is returned. If errors occurs, function return NULL.
 * 
 *  @param [in] in_size input size (width of input matrix)
 *  @param [in] b_size  batch size (heigh of input matrix)
 *  @param [in] nn_size     Numer Of Layers
 */
nn_array_t* nn_create(uint32_t in_size, uint32_t b_size, uint16_t nn_size, ...);

void nn_destroy(nn_array_t *nn);

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */