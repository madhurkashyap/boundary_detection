# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 06:34:18 2018

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
from AcousticDataGenerator import AcousticDataGenerator

logfile = prog+'.log'
rootlog = initlog(logfile,level=logging.DEBUG);

rootlog.info('Starting new session');
corpus = Timit(root=sys.argv[1])
corpus.split_validation();
adg = AcousticDataGenerator(corpus=corpus,output='sequence',
                            ctc_mode=True,mbatch_size=8);
adg.fit_train(n_samples=10);
trgen = adg.train_generator();
valgen = adg.valid_generator();
X,Y = next(trgen);
idx = adg.train_idxs[0]
print(corpus._corpusdf.iloc[idx]);
print("Sample CTC training output")
print(X['the_input'][0])
print(X['the_labels'][0])
print(X['input_length'][0])
print(X['label_length'][0])
print('Train shapes step1: '+str([x.shape for x in X.values()]))

X,Y = next(trgen);
idx = adg.train_idxs[8]
print(corpus._corpusdf.iloc[idx]);
print("Sample CTC training output")
print(X['the_input'][0])
print(X['the_labels'][0])
print(X['input_length'][0])
print(X['label_length'][0])
print('Train shapes step2: '+str([x.shape for x in X.values()]))
