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

def bidi_l2_ce(rnncell,input_dim,units1,units2,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();
    model.add(Bidirectional(rnncell(units1,
                                    return_sequences=True,
                                    dropout=before_dropout,
                                    recurrent_dropout=rec_dropout,
                                   ),
                            batch_input_shape=(None,None,input_dim)
                           ));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(Bidirectional(rnncell(units2,
                                    return_sequences=True,
                                    dropout=before_dropout,
                                    recurrent_dropout=rec_dropout,
                                   ),
                            batch_input_shape=(None,None,input_dim)
                           ));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))
    print(model.summary())
    return model


def bidi_l1_ce(rnncell, input_dim,units,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();
    model.add(Bidirectional(rnncell(units,
                                    return_sequences=True,
                                    dropout=before_dropout,
                                    recurrent_dropout=rec_dropout,
                                    ),
                            batch_input_shape=(None,None,input_dim),
                            ));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')));

    print(model.summary())
    return model

def uni_l1_ce(rnncell, input_dim,units,output_dim,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    model = Sequential();
    model.add(rnncell(units,
                      return_sequences=True,
                      dropout=before_dropout,
                      recurrent_dropout=rec_dropout,
                      batch_input_shape=(None,None,input_dim),
                     ));

    if batchnorm: model.add(BatchNormalization());
    if after_dropout>0: model.add(Dropout(after_dropout));

    model.add(TimeDistributed(Dense(output_dim,activation='softmax')))

    print(model.summary())
    return model

def bidi_lstm2(input_dim,units1,units2,output_dim,gpu=False,batchnorm=False,
                 before_dropout=0.0,after_dropout=0.0,rec_dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    return bidi_l2_ce(rnncell,units1,units2,output_dim,
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

def uni_gru_ctc(input_dim,units,output_dim,gpu=False,batchnorm=False,dropout=0.0):
    rnncell = CuDNNGRU if gpu else GRU;

    input_data = Input(name='the_input', shape=(None, input_dim))
    last = rnncell(output_dim, return_sequences=True, 
                    implementation=2, name='rnn')(input_data)

    if batchnorm: last = BatchNormalization()(last);
    if dropout>0: last = Dropout(dropout)(last);

    time_dense = TimeDistributed(Dense(output_dim))(last)

    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: x
    print(model.summary())

    return model

def uni_lstm_ctc(input_dim,units,output_dim,gpu=False,batchnorm=False,dropout=0.0):
    rnncell = CuDNNLSTM if gpu else LSTM;

    input_data = Input(name='the_input', shape=(None, input_dim))
    last  = rnncell(output_dim, return_sequences=True, 
                       implementation=2, name='rnn')(input_data)

    if batchnorm: last = BatchNormalization()(last);
    if dropout>0: last = Dropout(dropout)(last);

    time_dense = TimeDistributed(Dense(output_dim))(last)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

