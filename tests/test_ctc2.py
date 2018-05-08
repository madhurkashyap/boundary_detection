# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:55 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging
import numpy as np
from keras.optimizers import Adam

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import initlog
from SpeechCorpus import Timit
from AcousticModels import *
from TrainUtils import train_model
from AcousticDataGenerator import AcousticDataGenerator


corpus = Timit(root="C:\\Users\\nxa17016\\ML\\pyml\\RNN\\assignment3\\dataset")
corpus.split_validation();
corpus.report_statistics(folder='report/images',reptext=False);

# The sampling frequency reduction does not create any difference in output loss
#adg = AcousticDataGenerator(corpus=corpus,ctc_mode=True,mbatch_size=32,
#                            output='sequence',mode='grapheme',
#                            mfcc_win=0.0125, mfcc_step=0.005);

adg = AcousticDataGenerator(corpus=corpus,ctc_mode=True,mbatch_size=32,
                            output='sequence',mode='grapheme',
                            mfcc_win=0.025, mfcc_step=0.010,
                            mfcc_roc=True,mfcc_roa=True);

adg.fit_train(n_samples=100);
model = bidi_lstm_ctc(
        input_dim=adg.feature_dim,
        units=100,
        output_dim=adg.n_classes,
        gpu=False,
        batchnorm=True,
        after_dropout=0.0,
);
train_model(model,adg.train_generator(),adg.valid_generator(),
            'bidi_lstm_ctc',
           steps_per_epoch=10,
           validation_steps=adg.nb_valid,
           loss='ctc',
           optimizer=Adam(),
           epochs=1,
           save_period=0);

X,y = next(adg.valid_generator())
yp = model.predict(X);
print(''.join([adg.outmap[1][x] for x in X['the_labels'][1]]))
print(''.join([adg.outmap[1][x] for x in np.argmax(yp[1],axis=1)]))
