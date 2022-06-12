#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include "mx.h"
#include "nn.h"

typedef struct {
    void (* update)(void* opt_data, mx_t* weights, mx_t* delta, const nn_size idx);
    void (* params_destroy)(void* params);
    void *params;
}
optimizer_t;

#endif
