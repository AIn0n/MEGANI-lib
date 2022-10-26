#ifndef _IMAGE_LAYER_DATA_H_
#define _IMAGE_LAYER_DATA_H_
#include "MEGANI/mx/mx.h"

typedef struct {
	mx_size x, y, z;
} img_size_t;

typedef struct {
	img_size_t size;
	void *data;
} img_data_t;

#endif
