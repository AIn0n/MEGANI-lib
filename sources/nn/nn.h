#ifndef _NN_H_
#define _NN_H_
#include "mx.h"

//------------------------------------------------MACROS------------------------------------------

#define MAX(a, b)	((a) < (b) ? (b) : (a))
#define NO_FUNC ((act_func_t) {.func_cell =NULL, .func_mx =NULL})
#define RELU    ((act_func_t) {.func_cell =relu_deriv_cell, .func_mx =relu_mx})

/** @def NN_SIZE
 *  @brief Type for number of layer in neural network. By default it is uint16_t.
 */
#define NN_SIZE uint16_t

//------------------------------------------------STRUCTURES---------------------------------------

/** @brief Structure with all activations functions subroutines used in single layer.
 * 
 *  Structure consist of two function pointers - one of them is used in forwarding
 *  on whole matrix, second one is used in backpropagation as function derivative 
 *  with single matrix cell (function mx_hadam_lambda). I do this in that way to 
 *  do as small amount of operations on matrices as possible.
 * 
 *  @see mx_hadam_lambda
 */
typedef struct {
	MX_TYPE (*func_cell)(MX_TYPE);  /**< function used in forwarding */
	void (*func_mx)(mx_t *);        /**< function used in backpropagation */
}
act_func_t;

/** @brief Main neural network struct.
 * 
 *  This struct is main element of neural network in this library.
 *  It consist of all layers with data, values, outupt, etc (for more info see nl_t),
 *  number of them <len> and additional temporary matrix used in backpropagation and cnn, same for every
 *  layer for memory optimalization.
 * 
 *  @see nl_t
 */
typedef struct {
	struct nl_t		*layers; /**< all neurons layers in current network */
	mx_t			*temp; /**< temporary matrix shared between layers for things like im2col, value delta, etc */
	mx_t			*delta[2];
	MX_TYPE			alpha;  /**< alpha indicates learning speed */
	NN_SIZE			len;   /**< number of layers */
	MX_SIZE			in_len;
	MX_SIZE			batch_len;
}
nn_t;

/** @brief struct containing all data from single layer.
 * 
 *  This structure have every learnable and constant data from single layer.
 *  Cells like out, delta, data are universal for every layer, more focused structures,
 *  functions etc, based on layer purpose you can find in data field which is pointer
 *  to other structure (example: dense_data_t). Functions forward and backward are used
 *  in backpropagation.
 */
struct nl_t {
	mx_t* out;		/**< layer output */
	mx_t* weights;		/**< layer weights */
	void* data;		/**< layer specialized data */
	void (* free_data)	(void * data);
	/**< function used to free memory allocated for data */
	void (* forwarding)	(struct nl_t*, const mx_t*);	
	/**< function used in nn_predict() */
	void (* backwarding)	(struct nl_t *self, nn_t *n, const NN_SIZE idx, const mx_t *prev_out);
	/**< function use in nn_fit() */		
};

//------------------------------------------FUNCTIONS--------------------------------------------

nn_t* nn_create(const MX_SIZE in_len, const MX_SIZE batch_len, const MX_TYPE alpha);

/** @brief Free memory allocated for neural network struct.
 * 
 *  Frees memory which we allocated by nn_create() function.
 *  Functions is not thread safe.
 *  
 *  @param [in] nn network which we free
 */
void nn_destroy(nn_t *nn);

//TODO docs
void nn_predict(nn_t* nn, const mx_t* input);
void nn_fit(nn_t* nn, const mx_t *input, const mx_t* output);

//----------------------------------------ACTIVATION FUNCTIONS--------------------------------------------

void relu_mx(mx_t *a);
MX_TYPE relu_deriv_cell(MX_TYPE a);

#endif

/** @file nn.h
 *  @brief Header file for basic neural networks operations
 */

/** @mainpage MEGANI library documentation
 *  @section Introduction
 *  @subsection Goals
 *  This code have one simple goal: give programmer enough tools to build GAN
 *  without additional dependencies. The library is almost depedency-free and 
 *  platform independent for easy porting to other architectures, systems etc.
 * 
 *  @ref code conventions
 */
