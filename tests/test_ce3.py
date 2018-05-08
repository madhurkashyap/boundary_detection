# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:55 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging
import numpy as np
from functools import partial
from keras.optimizers import Adadelta
from sklearn.metrics import confusion_matrix

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import *
from PlotUtils import *
from SpeechCorpus import Timit
from AcousticModels import *
from TrainUtils import train_model,weighted_categorical_crossentropy
from AcousticDataGenerator import AcousticDataGenerator


#logfile = prog+'.log'
#rootlog = initlog(logfile,level=logging.DEBUG);

#rootlog.info('Starting new session');
if len(sys.argv)>1:
    corpus = Timit(root=sys.argv[1]);
else:
    corpus = Timit(root='C:/Users/nxa17016/ML/pyml/RNN/assignment3/dataset')
    
corpus.split_validation();
#rootlog.info(corpus.report_statistics(folder='report/images'));
adg = AcousticDataGenerator(corpus=corpus,mbatch_size=32,
        mfcc_win=0.0125,mfcc_step=0.005,
        ce_encoding_mode='best',
        mode='phoneme', model_silence=True);
adg.fit_train(n_samples=1000);
model = bidi_lstm(input_dim=adg.feature_dim,units=20,output_dim=adg.n_classes,
                batchnorm=True,after_dropout=0.0);

train_model(model,adg.train_generator(),adg.valid_generator(),'bidi_gru_20',
            epochs=1,steps_per_epoch=adg.nb_train-100,validation_steps=adg.nb_valid-10,
            verbose=1,save_period=0,optimizer=Adadelta(),report_stats=True,
            class_names=list(adg.outmap[0].keys()));