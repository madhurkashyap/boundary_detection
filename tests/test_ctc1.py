# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:15:55 2018

@author: Madhur Kashyap 2016EEZ8350
"""

import os
import sys
import logging
from keras.optimizers import Adam

prog = os.path.basename(__file__)
codedir = os.path.join(os.path.dirname(__file__),"..","code")
sys.path.append(codedir)

from Utils import initlog
from SpeechCorpus import Timit
from AcousticModels import bidi_ctc_lstm2
from TrainUtils import train_model
from AcousticDataGenerator import AcousticDataGenerator


corpus = Timit(root="C:\\Users\\nxa17016\\ML\\pyml\\RNN\\assignment3\\dataset")
corpus.split_validation();
corpus.report_statistics(folder='report/images',reptext=False);
adg = AcousticDataGenerator(corpus=corpus,ctc_mode=True,mbatch_size=32,
                            output='sequence',mode='grapheme',
                            mfcc_win=0.0125, mfcc_step=0.005);
adg.fit_train(n_samples=100);
model = bidi_ctc_lstm2(
        input_dim=adg.feature_dim,
        units1=100,
        units2=100,
        output_dim=adg.n_classes,
        gpu=False,
        batchnorm=True,
        after_dropout=0.0,
);
train_model(model,adg.train_generator(),adg.valid_generator(),
            'bidi_lstm2_ctc',
           steps_per_epoch=adg.nb_train,
           validation_steps=adg.nb_valid,
           loss='ctc',
           optimizer=Adam(),
           epochs=1,
           save_period=0);

X,y = next(adg.valid_generator())
print(model.predict(X));
