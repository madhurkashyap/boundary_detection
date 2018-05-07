# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:55 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import *
from PlotUtils import *
from SpeechCorpus import Timit
from AcousticModels import uni_gru
from TrainUtils import train_model
from AcousticDataGenerator import AcousticDataGenerator


#logfile = prog+'.log'
#rootlog = initlog(logfile,level=logging.DEBUG);

#rootlog.info('Starting new session');
if len(sys.argv)>1:
    corpus = Timit(root=sys.argv[1]);
else:
    corpus = Timit(root="C:/Users/nxa17016/ML/pyml/RNN/assignment3/dataset")
    
corpus.split_validation();
#rootlog.info(corpus.report_statistics(folder='report/images'));
adg = AcousticDataGenerator(corpus=corpus,mbatch_size=32,
        mfcc_win=0.0125,mfcc_step=0.010,
        ce_encoding_mode='best');
adg.fit_train(n_samples=100);
trgen = adg.train_generator();
model = uni_gru(input_dim=adg.feature_dim,units=10,output_dim=adg.n_classes,
                batchnorm=True,after_dropout=0.0);
train_model(model,adg.train_generator(),adg.valid_generator(),'uni_gru_200_mfcc',
            epochs=20,steps_per_epoch=adg.nb_train,validation_steps=adg.nb_valid,
            verbose=1,save_period=0);
