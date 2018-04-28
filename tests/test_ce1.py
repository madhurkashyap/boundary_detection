# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:55 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging
from keras.optimizers import SGD

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import *
from PlotUtils import *
from SpeechCorpus import Timit
from AcousticModels import uni_gru
from TrainUtils import train_model
from AcousticDataGenerator import AcousticDataGenerator


logfile = prog+'.log'
rootlog = initlog(logfile,level=logging.DEBUG);

rootlog.info('Starting new session');
corpus = Timit(root=sys.argv[1]);
corpus.split_validation();
rootlog.info(corpus.report_statistics(folder='report/images'));
adg = AcousticDataGenerator(corpus=corpus,mbatch_size=32);
adg.fit_train(n_samples=100);
trgen = adg.train_generator();
model = uni_gru(input_dim=adg.feature_dim,units=200,output_dim=adg.n_classes)
train_model(model,adg.train_generator(),adg.valid_generator(),'uni_gru_200',
            epochs=20,steps_per_epoch=100,verbose=1,save_period=5);
