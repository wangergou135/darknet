# Darknet-PoolFormer #
This is a project to rewrite poolformer on darknet.

step 0: List the layers the poolformer has, find out which layers darknet have already supported and which not;

step 1: Rewrite those layers which is not supported in darknet;

step 2: Write the poolformer in .cfg file 

# update log #
<font color=#00ffff>1. GroupNorm cpu version added</font>


<font color=#00ffff>2. DropPath cpu version soon</font>

# poolformer layers #

| layer-type    | supported |
| ------------  | --------: |  
| Convolution   |   true   |  
| Pooling   |   true   |  
| GroupNorm   |   false   |  
| ElementAdd   |   true   |  
| MLP   |   true   |  
| DropPath   |   false   |  
| LayerScale   |   false   |  





## Cite: 

* **source code - Pytorch :** https://github.com/sail-sg/poolformer

* **source code - Darknet:** https://github.com/pjreddie/darknet
