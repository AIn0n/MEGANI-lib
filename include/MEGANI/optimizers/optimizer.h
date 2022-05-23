#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include "mx.h"

typedef struct {
    void (* update)(mx_t*, mx_t*, void*);
    void *params;
}
optimizer_t;

#endif
