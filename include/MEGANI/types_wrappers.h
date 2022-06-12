#ifndef _TYPE_WRAPPERS_H_
#define _TYPE_WRAPPERS_H_
#include "stdint.h"

/* neural network number of layers size */
typedef uint16_t nn_size;

/** @def    mx_type
*   @brief  Type of variables which contain neural values.
*
*   By default it is "double".
*/
typedef double mx_type;

/** @def    NN_ZERO
 *  @brief  zero cast on mx_type.
 */
#define NN_ZERO ((mx_type)0)

/** @def mx_size
 *  @brief Type of variables which contain matrix size, x, y. 
 * 
 *  By default it is uint32_t. I made this macro 'cause many times I used many 
 *  diffrent types for same tasks like iteration over matrix etc.
 */
typedef uint32_t mx_size;

#endif
