# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 07:07:16 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import re
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    CuDNNGRU, CuDNNLSTM, Dropout, Flatten)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Basic uni and bi-directional layer definitions
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def is_cuda_layer(layer):
    return re.match('^.*CuDNN.*$',str(layer))

def bidi_layer(rnncell,input_dim,units,rec_dropout):
    iscud = is_cuda_layer(rnncell)
    if iscud:
        layer = Bidirectional(rnncell(units,return_sequences=True,),
                            batch_input_shape=(None,None,input_dim));
    else:
        layer = Bidirectional(rnncell(units,return_sequences=True,
                                      recurrent_dropout=rec_dropout),
                            batch_input_shape=(None,None,input_dim));
    return layer

def uni_layer(rnncell,input_dim,units,rec_dropout):
    iscud = is_cuda_layer(rnncell);
    if iscud:
        layer = rnncell(units,return_sequences=True,
                        batch_input_shape=(None,None,input_dim));
    else:
        layer = rnncell(units,return_sequences=True,
                        recurrent_dropout=rec_dropout,
                        batch_input_shape=(None,None,input_dim));
    return layer


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Generic bi-di and uni Network definitions
# Loss function - Cross entropy
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
def bidi_l2_ce(rnncell,input_dim,units1,units2,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();
    if before_dropout>0:
        model.add(Dropout(before_dropout,
                          batch_input_shape=(None,None,input_dim)));
    model.add(bidi_layer(rnncell,input_dim,units1,rec_dropout));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));
    
    model.add(bidi_layer(rnncell,input_dim,units2,rec_dropout));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    print(model.summary())

    return model


def bidi_l1_ce(rnncell, input_dim,units,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();

    if before_dropout>0:
        model.add(Dropout(before_dropout,
                          batch_input_shape=(None,None,input_dim)));
    model.add(bidi_layer(rnncell,input_dim,units,rec_dropout));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')));

    print(model.summary())
    return model

def uni_l1_ce(rnncell, input_dim,units,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();

    if before_dropout>0:
        model.add(Dropout(before_dropout,
                          batch_input_shape=(None,None,input_dim)));
    model.add(uni_layer(rnncell,input_dim,units,rec_dropout));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))

    print(model.summary())
    return model

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Finalized models for cross entropy loss function
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def bidi_lstm2(input_dim,units1,units2,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    return bidi_l2_ce(rnncell,input_dim,units1,units2,output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);

def bidi_gru2(input_dim,units1,units2,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNGRU if gpu else GRU;

    return bidi_l2_ce(rnncell,input_dim,units1,units2,output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);

def bidi_lstm(input_dim, units, output_dim,gpu=False,batchnorm=False,
         before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    return bidi_l1_ce(rnncell, input_dim, units, output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);

def bidi_gru(input_dim, units, output_dim,gpu=False,batchnorm=False,
            before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNGRU if gpu else GRU;

    return bidi_l1_ce(rnncell, input_dim, units, output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);

def uni_gru(input_dim, units, output_dim, gpu=False, batchnorm=False,
             before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNGRU if gpu else GRU;

    return uni_l1_ce(rnncell, input_dim, units, output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);

def uni_lstm(input_dim, units, output_dim, gpu=False, batchnorm=False,
              before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    return uni_l1_ce(rnncell, input_dim, units, output_dim,
                 batchnorm=batchnorm,
                 before_dropout=before_dropout,
                 after_dropout=after_dropout,
                 rec_dropout=rec_dropout);


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Finalized models for CTC loss function
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def bidi_ctc_lstm2(input_dim,units1,units2,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    
    rnncell = CuDNNLSTM if gpu else LSTM;
    
    input_data = Input(name='the_input', shape=(None, input_dim))
    last = input_data;
    if before_dropout>0:
        last = Dropout(before_dropout)(last);
        #batch_input_shape=(None,None,input_dim)));###
    
    # First bi-directional layer
    if gpu:
        last = Bidirectional(rnncell(units1,return_sequences=True))(last);
    else:
        last = Bidirectional(rnncell(units1,return_sequences=True,
                      recurrent_dropout=rec_dropout))(last);

    if batchnorm: last = BatchNormalization()(last);
    if after_dropout>0: last = Dropout(after_dropout)(last);

    # Second bi-directional layer
    if gpu:
        last = Bidirectional(rnncell(units2,return_sequences=True))(last);
    else:
        last = Bidirectional(rnncell(units2,return_sequences=True,
                      recurrent_dropout=rec_dropout))(last);

    if batchnorm: last = BatchNormalization()(last);
    if after_dropout>0: last = Dropout(after_dropout)(last);
    
    time_dense = TimeDistributed(Dense(output_dim))(last)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: x
    print(model.summary())
    return model
    
def bidi_lstm_ctc(input_dim,units,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,rec_dropout=0.0,after_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    input_data = Input(name='the_input', shape=(None, input_dim))
    last = input_data;
    if before_dropout>0: last = Dropout(before_dropout)(last);
    last  = Bidirectional(rnncell(output_dim, return_sequences=True), 
                                  name='bidi_rnn')(last)

    if batchnorm: last = BatchNormalization()(last);
    if after_dropout>0: last = Dropout(after_dropout)(last);

    time_dense = TimeDistributed(Dense(output_dim))(last)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
 
def uni_gru_ctc(input_dim,units,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,rec_dropout=0.0,after_dropout=0.0):
    rnncell = CuDNNGRU if gpu else GRU;

    input_data = Input(name='the_input', shape=(None, input_dim))
    last = input_data;
    if before_dropout>0: last = Dropout(before_dropout)(last);
    last = rnncell(output_dim, return_sequences=True, 
                    implementation=2, name='rnn')(last)

    if batchnorm: last = BatchNormalization()(last);
    if after_dropout>0: last = Dropout(after_dropout)(last);

    time_dense = TimeDistributed(Dense(output_dim))(last)

    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: x
    print(model.summary())

    return model

def uni_lstm_ctc(input_dim,units,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,rec_dropout=0.0,after_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    input_data = Input(name='the_input', shape=(None, input_dim))
    last = input_data;
    if before_dropout>0: last = Dropout(before_dropout)(last);
    last  = rnncell(output_dim, return_sequences=True, 
                       implementation=2, name='rnn')(last)

    if batchnorm: last = BatchNormalization()(last);
    if after_dropout>0: last = Dropout(after_dropout)(last);

    time_dense = TimeDistributed(Dense(output_dim))(last)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


