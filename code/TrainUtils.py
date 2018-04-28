# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:14:12 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import numpy as np
import keras.backend as K
from Utils import create_folder, dump_data
from itertools import product
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Lambda

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def ctc_func(args):
    y_pred, y_true, input_length, label_length = args
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def add_ctc(model):
    Y_pred = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(model.output_length)(input_lengths)
    ctc_loss = Lambda(ctc_func, output_shape=(1,), name='ctc')(
        [model.output, Y_pred, output_lengths, label_lengths])
    model = Model(
        inputs=[model.input, Y_pred, input_lengths, label_lengths], 
        outputs=ctc_loss)
    return model

def batch_temporal_categorical(y,n_classes):
    assert len(y.shape)==2, "Temporal batch predictions should be 2-dimensional"
    yoh = np.zeros((y.shape[0],y.shape[1],n_classes))
    for i in range(y.shape[0]):
        oh = [to_categorical(x,n_classes) for x in y[i]]
        yoh[i]=oh;
    return yoh;

def train_model(model,trgen,valgen,prefix,
                epochs=1,verbose=1,loss='binary_crossentropy',
                history_folder='./history',model_folder='./models',
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, 
                    nesterov=True, clipnorm=5),
                steps_per_epoch=100, validation_steps=10,
                metrics = ['acc']):

    model_path = os.path.join(model_folder,prefix+'.{epoch:02d}-{val_loss:.2f}.hdf5')
    pickle_path = os.path.join(history_folder,prefix+'.pkl');
    create_folder(model_folder); create_folder(history_folder);
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)
    if loss=='ctc':
        ctcmodel = add_ctc(model);
        ctcmodel.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
        hist = ctcmodel.fit_generator(generator=trgen,
                               validation_data=valgen,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,                           
                               callbacks=[checkpointer],
                               verbose=verbose);
    else:    
        model.compile(optimizer=optimizer,loss=loss,metrics=metrics);
        hist = model.fit_generator(generator=trgen,
                               validation_data=valgen,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,                           
                               callbacks=[checkpointer],
                               verbose=verbose);

    dump_data(hist.history,pickle_path);
    return hist
