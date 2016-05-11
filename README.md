# Identity Mappings in Deep Residual Networks in Lasagne/Theano

Reproduction of some of the results from the recent [MSRA ResNet](https://arxiv.org/abs/1603.05027) paper. Exploring the full-preactivation style residual layers. Luckily this paper was done on CIFAR-10 so I am able to recreate the results, my poor computer can't handle ResNets with ImageNet.

## Results

Results are presented as classification error percent just like in the paper.

| ResNet Type | Original Paper | My Results |
| -----------|-----------|----------- |
| ResNet-110 | 6.37 | 6.38 |
| ResNet-164 | 5.46 | 6.07 |

**Note:** ResNet-110 is the stacked 3x3 filter variant and ResNet-164 is the 'botttleneck' architecture. Both use the new pre-activation units as proposed in the paper.

### ResNet-110

![ResNet-110](http://i.imgur.com/Y7VrxOC.png)

### ResNet-164

![ResNet-164](http://i.imgur.com/Zg8fJvX.png)

## Implementation details

I am using a smaller batch size because of hardware constraints, 32 instead of 128. My data augmentation is exactly the same, only translations by padding then copping and left-right flipping.

## Pre-Trained weights

The weights of the trained networks are available for download. Weights are from an old try, may not load into current models in models.py. Will update when ResNet-164 matches results from paper.
