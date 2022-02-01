#include "droppath_layer.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>

droppath_layer make_droppath_layer(int batch, float probability)
{
    droppath_layer l = {0};
    l.type = DROPPATH;
    l.probability = probability;
    l.inputs = inputs;
    l.outputs = inputs;
    l.batch = batch;
    l.rand = calloc(batch, sizeof(float));
    l.scale = 1./(1.-probability);
    l.forward = forward_droppath_layer;
    l.backward = backward_droppath_layer;
    #ifdef GPU
    l.forward_gpu = forward_droppath_layer_gpu;
    l.backward_gpu = backward_droppath_layer_gpu;
    l.rand_gpu = cuda_make_array(l.rand, inputs*batch);
    #endif
    fprintf(stderr, "dropout       p = %.2f               %4d  ->  %4d\n", probability, inputs, inputs);
    return l;
} 

void resize_droppath_layer(droppath_layer *l, int inputs)
{
    l->rand = realloc(l->rand, l->inputs*l->batch*sizeof(float));
    #ifdef GPU
    cuda_free(l->rand_gpu);

    l->rand_gpu = cuda_make_array(l->rand, inputs*l->batch);
    #endif
}

void forward_droppath_layer(droppath_layer l, network net)
{
    int i;
    if (!net.train) return;
    for(i = 0; i < l.batch; ++i){
        float r = rand_uniform(0, 1);
        l.rand[i] = r;
        
        if(r < l.probability)  
            fill_cpu(l.inputs, 0, net.input[i * l.inputs], 1);
        else 
            scal_cpu(l.inputs, l.scale, net.input[i * l.inputs], 1);
    }
}

void backward_droppath_layer(droppath_layer l, network net)
{
    int i;
    if(!net.delta) return;
    for(i = 0; i < l.batch; ++i){
        float r = l.rand[i];
        if(r < l.probability) 
            fill_cpu(l.inputs, 0, net.delta[i * l.inputs], 1);
        else 
            scal_cpu(l.inputs, l.scale, net.delta[i * l.inputs], 1);
    }
}

