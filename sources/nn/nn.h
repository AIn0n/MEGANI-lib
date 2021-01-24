#ifndef _NN_H_
#define _NN_H
#include "matrix.h"


/** @struct nn_layer
 *  @brief Structure with single neural network layer.
 * 
 *  I designed this struct as small and fast as possible.
 *  It depend matrix.h file for structs and functions for
 *  matrix.
 * 
 *  @see mx_t 
 */
typedef struct {
    mx_t out;           /**< matrix with layer output */
    mx_t val;           /**< values of neurons in current layer */
    mx_t delta;         /**< delta, used in backpropagation */
    mx_t drop;          /**< dropout mask */
    NN_TYPE drop_rate;  /**< percentage amount of turned off neurons in dropout */
    void (*activ_func)(NN_TYPE*, uint8_t);  /**< activation function pointer */
} nn_layer;

typedef nn_layer* nn_array;

/** @brief Function creates neural network structure.
 * 
 *  Function create empty neural network without any layer,
 *  it is just allocation for single NULL pointer, other funcs
 *  can adds new layers. Function is thread-safe if stdlib calloc
 *  is thread-safe.
 * 
 *  On success, pointer to structure is returned. If errors occurs, function return NULL.
 */
nn_array* nn_create(void);

/** @brief Function adds new layer to existing network.
 * 
 *          
 *              TODO
 * 
 */
uint8_t nn_add_layer(
nn_array*   nn, 
uint32_t    neurons, 
uint32_t    in_size,
void        (*activ_func)(NN_TYPE*, uint8_t),
uint32_t    batch_size);
#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */