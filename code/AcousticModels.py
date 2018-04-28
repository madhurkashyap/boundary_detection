# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 07:07:16 2018

@author: Madhur Kashyap 2016EEZ8350
"""

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    CuDNNGRU, CuDNNLSTM, Dropout, Flatten)

def bidi_lstm2(input_dim,units1,units2,output_dim,gpu=False):
    model = Sequential();
    if gpu:
        model.add(Bidirectional(CuDNNLSTM(units1, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))
        model.add(Bidirectional(CuDNNLSTM(units2, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))
    else:
        model.add(Bidirectional(LSTM(units1, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))
        model.add(Bidirectional(LSTM(units2, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))
    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    # Allow every sample to have different length for ctc
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidi_gru(input_dim, units, output_dim,gpu=False):
    model = Sequential();
    if gpu:
        model.add(Bidirectional(CuDNNGRU(units, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))
    else:
        model.add(Bidirectional(GRU(units, return_sequences=True),
                               batch_input_shape=(None,None,input_dim)))

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    # Allow every sample to have different length for ctc
    model.output_length = lambda x: x
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
    # Allow every sample to have different length for ctc
    model.output_length = lambda x: x
    print(model.summary())
    return model

def uni_gru_ctc(input_dim,units,output_dim,gpu=False):
    input_data = Input(name='the_input', shape=(None, input_dim))
    if gpu:
        simp_rnn = CuDNNGRU(output_dim, return_sequences=True, 
                       implementation=2, name='rnn')(input_data)
    else:
        simp_rnn = GRU(output_dim, return_sequences=True, 
                       implementation=2, name='rnn')(input_data)
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
