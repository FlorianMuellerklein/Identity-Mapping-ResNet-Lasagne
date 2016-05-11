import os
import gzip
import time
import pickle
import datetime
import random
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from models import ResNet_FullPreActivation, ResNet_BottleNeck_FullPreActivation
from utils import load_pickle_data_test

BATCHSIZE = 1

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
output_layer = ResNet_BottleNeck_FullPreActivation(X)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

output_class = T.argmax(output_test, axis=1)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)
predict_class = theano.function(inputs=[X], outputs=output_class)

'''
Load data and make predictions
'''
test_X, test_y = load_pickle_data_test()

# load network weights
f = gzip.open('data/weights/resnet164_fullpreactivation.pklz', 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)

#make predictions
pred_labels = []
for j in range((test_X.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
    sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
    X_batch = test_X[sl]
    pred_labels.extend(predict_class(X_batch))

pred_labels = np.array(pred_labels)
print pred_labels.shape

'''
Compare differences
'''
same = 0
for i in range(pred_labels.shape[0]):
    if test_y[i] == pred_labels[i]:
        same += 1

print('Percent same, ', (float(same) / float(pred_labels.shape[0])))
