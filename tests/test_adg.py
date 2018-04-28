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
if len(sys.argv)>1:
    corpus = Timit(root=sys.argv[1])
else:
    corpus = Timit(root='C:/Users/nxa17016/ML/pyml/RNN/assignment3/dataset')
corpus.split_validation();
adg = AcousticDataGenerator(corpus=corpus,mbatch_size=32);
adg.fit_train(n_samples=10);
trgen = adg.train_generator();
for i in range(113):
    print ("INFO: Generator cycle %d"%i);
    X,Y = next(trgen);
