#ifndef _NN_H_
#define _NN_H_
#include "mx.h"
#include "optimizer.h"
#include "types_wrappers.h"

//------------------------------------------------MACROS------------------------------------------

#define MAX(a, b)	((a) < (b) ? (b) : (a))
#define NO_FUNC ((act_func_t) {.func_cell =NULL, .func_mx =NULL})
#define RELU    ((act_func_t) {.func_cell =relu_deriv_cell, .func_mx =relu_mx})

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
	mx_type (* func_cell)(mx_type);	/**< function used in forwarding */
	void 	(* func_mx)(mx_t *);	/**< function used in backpropagation */
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
	mx_t			*delta[2]; /**< Two delta matrices to switch between in backpropagation */
	optimizer_t		optimizer; /**< Optimizer structure with optimize function and parameters */
	nn_size			len;   /**< number of layers */
	mx_size			in_len;	/**< size of input */
	mx_size			batch_len; /**< number of batches */
	uint8_t			error;	/**< Error code indicating that something was done wrong */
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
	mx_t *out;		/**< layer output */
	mx_t *weights;		/**< layer weights */
	void *data;		/**< layer specialized data */
	void (* free_data)	(void *);	/**< function to free layer specific data */
	/**< function used to free memory allocated for data */
	void (* forwarding)	(struct nl_t*, const mx_t*);
	/**< function used in nn_predict() */
	void (* backwarding)	(const nn_t *, const nn_size, const nn_size, const mx_t *);
	/**< function use in nn_fit() */
};

//------------------------------------------FUNCTIONS--------------------------------------------

nn_t* nn_create(const mx_size in_len, const mx_size batch_len);

/** @brief Free memory allocated for neural network struct.
 * 
 *  Frees memory which we allocated by nn_create() function.
 *  Functions is not thread safe.
 *  
 *  @param [in] nn network which we free
 */
void nn_destroy(nn_t *nn);

/**
 * @brief predict network response based on input
 * 
 * @param [in] nn network which response we want to get 
 * @param [in] input input with proper size and batch length
 */
void nn_predict(nn_t *nn, const mx_t *input);

/**
 * @brief learn network with input and expected output
 * 
 * @param [in] nn network which we want to learn
 * @param [in] input network with proper size and batch length
 * @param [in] output exepcted output with size equal to last layer output size
 */
void nn_fit(nn_t *nn, const mx_t *input, const mx_t *output);

//----------------------------------------ACTIVATION FUNCTIONS--------------------------------------------

void relu_mx(mx_t *a);
mx_type relu_deriv_cell(mx_type a);

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
