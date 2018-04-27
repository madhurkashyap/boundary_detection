# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 07:07:16 2018

@author: Madhur Kashyap 2016EEZ8350
"""

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    CuDNNGRU, Dropout, Flatten)

def bidi_gru(input_dim, units, output_dim,gpu=False):
    model = Sequential();
    if gpu:
        model.add(Bidirectional(CuDNNGRU(units, return_sequences=True,
                                batch_input_shape=(None,None,input_dim))))
    else:
        model.add(Bidirectional(GRU(units, return_sequences=True,
                                batch_input_shape=(None,None,input_dim))))

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    print(model.summary())
    return model

def uni_gru(input_dim, units, output_dim, gpu=False):
    model = Sequential();
    if gpu:
        model.add(CuDNNGRU(units, return_sequences=True,
                           batch_input_shape=(None,None,input_dim)))
    else:
        model.add(GRU(units, return_sequences=True,
                           batch_input_shape=(None,None,input_dim)))
    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    print(model.summary())
    return model
