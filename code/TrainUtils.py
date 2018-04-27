# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:14:12 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import numpy as np
import keras.backend as K
from itertools import product
from keras.utils import to_categorical

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def batch_temporal_categorical(y,n_classes):
    assert len(y.shape)==2, "Temporal batch predictions should be 2-dimensional"
    yoh = np.zeros((y.shape[0],y.shape[1],n_classes))
    for i in range(y.shape[0]):
        oh = [to_categorical(x,n_classes) for x in y[i]]
        yoh[i]=oh;
    return yoh;
