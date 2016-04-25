# Identity Mappings in Deep Residual Networks in Lasagne/Theano

Reproduction of some of the results from the recent [MSRA ResNet](https://arxiv.org/abs/1603.05027) paper. Exploring the full-preactivation style residual layers. Luckily this paper was done on CIFAR-10 so I am able to recreate the results, my poor computer can't handle ResNets with ImageNet. The ResNet-110 network's results were reproduced first try within 0.01% which was great!

## Results

| ResNet Type | Original Paper | My Results |
| -----------|-----------|----------- |
| ResNet-110 | 6.37 | 6.38 |
| ResNet-164 | 5.46 | Still Running |

Note - ResNet-110 is the stacked 3x3 filter variant and ResNet-164 is the 'botttleneck' architecture. Both use the new pre-activation units as proposed in the paper.

## Implementation details

There are a few small differences in my implementation and that of the paper. I am using a smaller batch size because of hardware constraints, 32 instead of 128. I am also using a lower initial learning rate 0.01 instead of 0.1, I find it to be more stable. My data augmentation is exactly the same, only translations by padding then copping and left-right flipping.

An added benefit is that these changes show that the ResNet architecture is robust to small hyperparameter tweaks.

## Pre-Trained weights

The weights of the trained networks are available for download.
