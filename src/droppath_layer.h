#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer droppath_layer;

droppath_layer make_droppath_layer(int batch, int inputs, float probability);

void forward_droppath_layer(droppath_layer l, network net);
void backward_droppath_layer(droppath_layer l, network net);
void resize_droppath_layer(droppath_layer *l, int inputs);

#ifdef GPU
void forward_droppath_layer_gpu(droppath_layer l, network net);
void backward_droppath_layer_gpu(droppath_layer l, network net);

#endif
#endif
