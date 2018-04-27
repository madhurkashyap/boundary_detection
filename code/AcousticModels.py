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
from keras.optimizers import SGD


def bidi_gru(input_dim, units, output_dim):
    model = Sequential();
    model.add(Bidirectional(GRU(units, return_sequences=True,
                                batch_input_shape=(None,None,input_dim))))
    model.add(Dense(output_dim,activation='softmax'))
    #model.output_length = lambda x: x
    print(model.summary())
    return model

    
def gru(input_dim, units, output_dim):
    model = Sequential();
    model.add(GRU(units, return_sequences=True,
                                batch_input_shape=(None,None,input_dim)))

    # https://github.com/keras-team/keras/issues/3009
    #model.add(Flatten());
    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    model.output_length = lambda x: x
    print(model.summary())
    return model