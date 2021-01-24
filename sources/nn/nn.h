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
} nn_layer;

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */