# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 08:14:12 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import numpy as np
import keras.backend as K
from itertools import product
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.losses import binary_crossentropy

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

def train_model(model,save_path,trgen,valgen,
                epochs=1,verbose=1,ctc_mode=False,
                loss=binary_crossentropy,
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, 
                    nesterov=True, clipnorm=5),
                steps_per_epoch=100, validation_steps=10,
                metrics = ['acc']):

    if ctc_mode:
        raise ValueError("CTC based model training is not yet implemented")

    checkpointer = ModelCheckpoint(filepath=save_path, verbose=1)
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics);
    hist = model.fit_generator(generator=trgen,
                               validation_data=valgen,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch,
                               validation_steps=validation_steps,                           
                               callbacks=[checkpointer],
                               verbose=verbose);
    return hist
